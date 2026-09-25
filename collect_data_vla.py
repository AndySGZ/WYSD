# -*- coding: utf-8 -*-
"""把抓取示范录成 VLA（pi0 / SmolVLA）能直接吃的 **LeRobotDataset**。

为什么单独一个脚本：π0/SmolVLA 这类 VLA 的输入是"图像 + 语言 + 本体状态"，LeRobot 的
训练入口只认 LeRobotDataset（parquet + 视频/图像 + meta）。原来 `data/grasp/data.npz`
是 17 维纯状态的轻量格式，只能喂本仓库的小 MLP，VLA 用不了。

数据由**脚本专家**（src/expert_grasp.py）生成 —— 不需要人遥操作，几十秒能出几百段；
要混入人手遥操作示范，用 manual_collect.py 加 `--images`（同为 LeRobotDataset）。

录到的东西（默认）：
    observation.images.top      (256,256,3)  场景相机（俯视桌面，看得到方块和蓝盘）
    observation.images.wrist    (256,256,3)  腕部相机（跟着夹爪，看得到齿和方块）
    observation.state           (7,)         本体感知：关节1..6 + 夹爪角
    action                      (7,)         目标关节1..6 + 夹爪命令
    task                        str          语言指令

关键约定（与仓库其它部分一致，见 src/expert_grasp.py 文件头）：
    action = **未来 lookahead(默认4) 拍之后的实测状态**（1 拍 = 50ms）。
    对动作分块式的 VLA（pi0/SmolVLA 都预测一整个 chunk）来说这正好是一条未来轨迹。
    状态里**不含**物块的位姿，所以策略必须真的看图（--privileged_state 才加进去做对照）。

用法：
    python collect_data_vla.py --episodes 50                     # -> data/grasp_vla/
    python collect_data_vla.py --episodes 200 --image_size 224 --overwrite
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
from src.env_grasp import UranusGraspEnv
from src.expert_grasp import pick_place

DEFAULT_TASK = "put the yellow cube on the blue disc"


def main(args):
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.overwrite:
            print(f"！{out_root} 已存在。要重录就加 --overwrite（会整个删掉），"
                  f"或者换 --out_root。")
            return 1
        shutil.rmtree(out_root)
    # 注意：这里不能预先 mkdir —— LeRobotDataset.create 要求目录不存在，它自己建

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print("！这个环境里没有可用的 lerobot.datasets（VLA 数据格式依赖它）：", e)
        print("  可用环境：C:\\Users\\sgz\\anaconda3\\envs\\lerobot_mujoco_vla_tutorial")
        return 1

    env = UranusGraspEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    size = args.image_size

    features = {
        "observation.images.top": {"dtype": "image", "shape": (size, size, 3),
                                   "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "image", "shape": (size, size, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    }
    if args.privileged_state:
        # 对照实验用：把物块/放置点相对齿面中点的位置也给策略（VLA 就该只靠图像）
        features["observation.environment_state"] = {
            "dtype": "float32", "shape": (6,), "names": ["env_state"]}

    ds = LeRobotDataset.create(repo_id=args.repo_id, root=out_root,
                               robot_type="uranus", fps=args.fps,
                               features=features,
                               image_writer_threads=args.writer_threads,
                               image_writer_processes=args.writer_processes)

    renderer = mujoco.Renderer(env.model, size, size)
    frame_imgs = []

    def grab_images():
        """渲染两路相机（与 stream 同步调用，所以下标一一对应）。"""
        imgs = {}
        for key, cam in (("observation.images.top", args.top_cam),
                         ("observation.images.wrist", args.wrist_cam)):
            renderer.update_scene(env.data, camera=cam)
            imgs[key] = renderer.render().copy()
        frame_imgs.append(imgs)

    n_ok, n_try, n_frames = 0, 0, 0
    t0 = time.time()
    print(f"输出：{out_root}   目标 {args.episodes} 段   fps={args.fps}   图像 {size}x{size}"
          f"   task='{args.task}'")

    while n_ok < args.episodes and n_try < args.max_tries:
        n_try += 1
        obj_pose, place = env.sample_layout(rng)
        frame_imgs.clear()
        res = pick_place(env, obj_pose=obj_pose, place_xyz=place,
                         record=True, rec_every=args.rec_every, lookahead=args.lookahead,
                         frame_hook=grab_images)
        if not res["ok"]:
            print(f"  [{n_try}] 专家没成功（抬升 {res['lift'] * 1000:.0f}mm），丢弃")
            continue

        # res["frames"][i] = (obs_i, [未来第 lookahead 拍的关节(6), 该拍夹爪命令(1)])
        # frame_imgs[i] 与它一一对应（同一时刻渲染的图）
        n_add = 0
        for i, (obs, act) in enumerate(res["frames"]):
            if i >= len(frame_imgs):
                break
            frame = {
                "observation.state": np.concatenate([obs[:6], obs[16:17]]).astype(np.float32),
                "action": act.astype(np.float32),
                "task": args.task,
                **{k: v for k, v in frame_imgs[i].items()},
            }
            if args.privileged_state:
                frame["observation.environment_state"] = np.concatenate(
                    [obs[6:9], obs[13:16]]).astype(np.float32)   # 物块相对位置 + 放置点相对位置
            ds.add_frame(frame)
            n_add += 1
        ds.save_episode()
        n_ok += 1
        n_frames += n_add
        if n_ok % args.print_every == 0 or n_ok == args.episodes:
            print(f"  collected {n_ok}/{args.episodes} 段 ({n_frames} 帧, try={n_try}, "
                  f"{time.time() - t0:.0f}s)")

    renderer.close()
    print(f"\n完成：{n_ok} 段 / {n_frames} 帧 -> {out_root}（LeRobotDataset，repo_id={args.repo_id}）")
    print(f"  专家成功率 {n_ok}/{n_try} = {n_ok / max(n_try, 1):.1%}，耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="data/grasp_vla")
    parser.add_argument("--repo_id", type=str, default="uranus_grasp_vla")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_tries", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20, help="控制频率（rec_every=25 步 * 2ms）")
    parser.add_argument("--rec_every", type=int, default=25, help="每多少个物理步录一帧")
    parser.add_argument("--lookahead", type=int, default=4, help="动作取未来第几拍（0.2s）")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--writer_threads", type=int, default=4,
                        help="LeRobotDataset 写图像的线程数")
    parser.add_argument("--writer_processes", type=int, default=4,
                        help="LeRobotDataset 写图像的进程数（写盘是采集的瓶颈，多进程能明显加速）")
    parser.add_argument("--top_cam", type=str, default="grasp_cam")
    parser.add_argument("--wrist_cam", type=str, default="wrist_cam")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--privileged_state", action="store_true",
                        help="额外把物块/放置点相对位置写进 observation.environment_state（对照实验）")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    sys.exit(main(parser.parse_args()))
