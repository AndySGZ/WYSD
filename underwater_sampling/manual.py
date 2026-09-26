# -*- coding: utf-8 -*-
"""采样任务族的人手操作采集入口（**不提供专家策略**，数据靠人操作）。

复用 `Lerobot-Uranus-VLA-Demo/manual_collect.py` 的那套键控笛卡尔遥操作 + 自动判成功 +
分段录制：只覆写"本段怎么布置/怎么算完成/显示什么"这三处，所以存盘格式与现有数据一致。

    python -m underwater_sampling.manual --level l1_rock_collect --episodes 20
    python -m underwater_sampling.manual --list

操作（沿用底座键位）：
    W/S 前后   A/D 左右   Q/E 升降   Space 夹爪开合   [ ] 调速
    T 同布局重来   R 换新布局   Enter 保存本段   Backspace 放弃   Esc 退出
    **C 切换水下/清水渲染**（默认按等级建议的水况显示水下图，就是策略将看到的画面）

⚠️ 本机械臂的操作包线（实测，决定任务怎么做）：
  - 只能平移齿面中点 + 开合夹爪，**腕部姿态是固定的**（6 自由度 IK 把齿条钉成"长轴沿 x、下俯 10°"）；
  - link6_mount 是个 6cm 直径、9.7cm 长的圆柱，其下缘比齿面中点还低约 1.5cm，
    所以**容器必须朝机器人一侧开口**，否则下探时腕部会啃壁（见 scenes/l1 的注释）；
  - 齿面中点 z 下限 0.365（LIMITS），齿面张开时两齿间距 7.7cm。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .env import _DEMO_DIR, SamplingEnv
from .tasks import LEVEL_ORDER, SURFACE_Z, describe_all, get

if _DEMO_DIR is None:
    raise ImportError("找不到 Lerobot-Uranus-VLA-Demo，无法复用遥操作底座。")
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import manual_collect                                        # noqa: E402
from manual_collect import TeleopCollector, TkView, build_frames   # noqa: E402

# 水下渲染开关（视图与采集器通过这个字典通信，省得互相持有引用）
WATER = {"on": True, "params": "coastal"}


class UnderwaterTkView(TkView):
    """把底座 TkView 的渲染换成水下成像模组。

    操作者看到的画面 = 策略将来看到的画面，这对"水下 VLA 的人手采集"很重要：
    否则人会照着清水图去对位，采出来的动作在水下根本看不清。
    """

    def __init__(self, model, width: int, height: int, cam, title="Uranus 水下采样采集"):
        super().__init__(model, width, height, cam, title=title)
        from underwater_vision import WaterRenderer

        offw, offh = int(model.vis.global_.offwidth), int(model.vis.global_.offheight)
        w, h = min(width, offw), min(height, offh)
        self.wr = WaterRenderer(model, w, h, params=WATER["params"], seed=0, use_depth=True)

    def draw(self, data, hud_lines):
        from PIL import Image, ImageDraw, ImageTk

        if WATER["on"]:
            img = Image.fromarray(self.wr.render(data, camera=self.cam))
        else:
            self.renderer.update_scene(data, camera=self.cam)
            img = Image.fromarray(self.renderer.render())
        d = ImageDraw.Draw(img, "RGBA")
        for i, line in enumerate(hud_lines):
            y = 8 + 24 * i
            w = d.textlength(line, font=self.font)
            d.rectangle([6, y, 16 + w, y + 22], fill=(0, 0, 0, 150))
            d.text((10, y + 2), line, font=self.font, fill=(255, 255, 255, 255))
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        self.root.update()


class SamplingCollector(TeleopCollector):
    """把底座的 pick-place 采集器改成"分级采样任务"版。"""

    # ---- 本段布置 ----
    def new_episode(self, obj_pose=None, place=None, layout=None):
        env = self.env
        if layout is None:
            # 底座 'T 同布局重来' 会走 new_episode(obj_pose=self.last_layout[0], ...)，
            # 而采样语义下 last_layout[0] 就是布局字典本身，这里认出来复用。
            layout = obj_pose if isinstance(obj_pose, dict) else env.sample_layout(self.rng)
        env.reset(layout)
        self.layout = layout
        self.last_layout = (layout, env.get_place_target().copy())

        self.p_target = env.tooth_midpoint().copy()
        self.home_target = self.p_target.copy()
        self.gripper_open = True
        self.jaw_cmd = float(env.open_angle)
        self.stream = []
        self.step_count = 0
        self.recording = False
        self.ep_obj0 = env.get_object_pos().copy()
        self.ep_place = env.get_place_target().copy()
        self.ep_peaks = [self.ep_obj0[2]]
        self.ep_saved = False
        self.success_since = None
        self.ep_frames = 0
        self.ep_travel = 0.0
        self._prev_p = None

    # ---- 本段算不算完成 ----
    def is_placed(self) -> bool:
        """用等级自己的判据：即时判据 + "真的停稳了"判据。

        不套用底座的"抬升 > 2cm"：L5/L6/L7 本来就不靠抬升，套上去会把有效段全判失败。
        """
        return bool(self.env.success() and self.env.settled())

    def control_tick(self, target=None):
        out = super().control_tick(target)
        # 每控制拍喂一次过程量（峰值夹持力、里程碑、下探深度）
        self.env.observe(self.args.ctrl_steps * self.env.model.opt.timestep)
        return out

    def handle_key(self, key: str):
        if key in ("c", "C"):
            WATER["on"] = not WATER["on"]
            self.message = f"水下渲染 {'开' if WATER['on'] else '关（清水）'}"
            print("  " + self.message, flush=True)
            return
        return super().handle_key(key)

    def finish_episode(self, keep: bool, auto: bool = False):
        env = self.env
        ok = self.is_placed()
        m = env.metrics()
        why = "" if ok else ("没进目标位" if not m["success"] else "进了但没停稳/翻倒")
        print(f"  本段：指令「{m['instruction']}」  obj→goal {m['obj_goal_dist'] * 1000:.0f}mm  "
              f"倾角 {m['tilt_deg']:.1f}°  峰值夹持力 {m['peak_force']:.0f}N  "
              f"工具走了 {self.ep_travel * 100:.0f}cm  录到 {len(self.stream)} 帧  "
              f"success={m['success']} settled={m['settled']}  "
              f"-> {'成功' if ok else '未完成：' + why}{'（自动）' if auto else ''}", flush=True)
        if keep and self.stream:
            if not ok and not self.args.keep_failed:
                print("  （没完成，不写进数据集；想强制保存加 --keep_failed）", flush=True)
                self.new_episode()
                return
            frames = build_frames(self.stream, self.args.lookahead)
            self.last_frames = frames
            self.last_episode = (env.info(), env.layout, frames)
            for obs, act in frames:
                self.obs_list.append(obs)
                self.act_list.append(act)
            self.ep_ends.append(len(self.obs_list))
            self.saved_episodes += 1
            self.ep_saved = True
            self.save()
            print(f"  已保存：episode {self.saved_episodes}/{self.args.episodes}，{len(frames)} 帧，"
                  f"累计 {len(self.obs_list)} 帧 -> {self.out_path()}", flush=True)
            if self.saved_episodes >= self.args.episodes:
                print(f"\n已经录满 {self.args.episodes} 段，自动退出。数据在 {self.out_path()}")
                self.quit = True
                return
        self.new_episode()

    def hud(self):
        env = self.env
        m = env.metrics()
        spec = env.spec
        tip = "完成！正在自动保存…" if self.is_placed() else spec["difficulty"]
        return [
            f"[{spec['title']}]  {spec['object_desc']} → {spec['goal_desc']}",
            f"指令：{m['instruction']}",
            f"已保存 {self.saved_episodes}/{self.args.episodes} 段  累计 {len(self.obs_list)} 帧  "
            f"速度 {self.speed:.2f}m/s  夹爪 {'张开' if self.gripper_open else '合上'}  "
            f"水下渲染 {'开' if WATER['on'] else '关'}",
            f"obj→goal {m['obj_goal_dist'] * 1000:.0f}mm  倾角 {m['tilt_deg']:.1f}°  "
            f"力 {m['force_grasp']:.0f}N  保持判据 {spec['hold_s']:.1f}s   {tip}",
            "W/S 前后  A/D 左右  Q/E 升降  Space 夹爪  [ ] 调速  C 水下/清水",
            "T 同布局重来  R 换新布局  Enter 保存  Backspace 放弃  Esc 退出",
        ]


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="采样任务族：人手操作采集")
    ap.add_argument("--level", type=str, default="l1_rock_collect", choices=LEVEL_ORDER)
    ap.add_argument("--list", action="store_true", help="打印难度阶梯表后退出")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--water", type=str, default=None,
                    help="覆盖该等级建议的水况（underwater_vision 预设名，如 turbid）")
    ap.add_argument("--no_water", action="store_true", help="不开水下渲染（清水对照）")
    # 以下沿用底座默认值，保证与现有采集链路口径一致
    ap.add_argument("--ctrl_steps", type=int, default=25)
    ap.add_argument("--rec_every", type=int, default=25)
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--speed", type=float, default=0.08, help="齿面中点移动速度 m/s")
    ap.add_argument("--hold", type=float, default=0.12)
    ap.add_argument("--ik_iters", type=int, default=12)
    ap.add_argument("--jaw_rate", type=float, default=0.10)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=650)
    ap.add_argument("--open_margin", type=float, default=0.030)
    ap.add_argument("--auto_success", dest="auto_success", action="store_true", default=True)
    ap.add_argument("--no_auto_success", dest="auto_success", action="store_false")
    ap.add_argument("--auto_hold", type=float, default=None,
                    help="判据需连续成立多久才算完成（默认用该等级的 hold_s）")
    ap.add_argument("--min_frames", type=int, default=20)
    ap.add_argument("--keep_failed", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    if args.list:
        print("采样任务族难度阶梯：\n")
        print(describe_all())
        print("\n每个等级的场景细节：WYSD/underwater_sampling/README.md")
        return 0

    spec = get(args.level)
    WATER["params"] = args.water or spec["water"]
    WATER["on"] = not args.no_water
    if args.auto_hold is None:
        args.auto_hold = float(spec["hold_s"])
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent / "data" / args.level)

    env = SamplingEnv(args.level, seed=args.seed)
    print(f"等级：{spec['title']}  ({args.level})")
    print(f"  难度轴：{spec['difficulty']}")
    print(f"  场景：{spec['xml']}    水下渲染：{'开' if WATER['on'] else '关'} / {WATER['params']}")
    print(f"  输出：{args.out_dir}    目标 {args.episodes} 段")

    # 底座 run() 里是全局查表 TkView，这里换成带水下渲染的视图（只改这一处）
    manual_collect.TkView = UnderwaterTkView

    col = SamplingCollector(env, args)
    if args.overwrite and Path(args.out_dir, "data.npz").exists():
        Path(args.out_dir, "data.npz").unlink()
    col.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
