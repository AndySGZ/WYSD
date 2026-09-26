# -*- coding: utf-8 -*-
"""报告用动画生成器：一张动画里放多路相机视角 + 中文注解。

    python -m underwater_vision.animate --all                 # 三张动画（推荐）
    python -m underwater_vision.animate --anim task           # 三视角抓取-放置
    python -m underwater_vision.animate --anim water          # 同一动作 · 不同水况
    python -m underwater_vision.animate --anim light          # 同一动作 · 不同光照
    python -m underwater_vision.animate --list

运动源：UranUS 抓取-放置**专家轨迹**（`WYSD/Lerobot-Uranus-VLA-Demo` 的脚本状态机）。
只在采集阶段跑一次轨迹、把每个控制拍（50ms）的 qpos 存下来，之后**只做正运动学 + 渲染**，
所以三张动画的运动完全一致 —— 变量只有水况/机位/光照，对比才站得住。
专家不可用时自动退回"程序化小幅摆动"（图能出，但语义弱，会打印警告）。

输出：`out/anim/<名字>.gif` 与 `.mp4`（GIF 用 PIL，MP4 用 cv2/mp4v）。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .params import WaterParams
from .paths import ENV_VAR, find_demo_dir
from .render import WaterRenderer
from .scene import SceneStyle

_HERE = Path(__file__).resolve().parent
# demo 仓库位置向上逐层探测（兼容本模块放在仓库外 / 仓库根内两种位置）
_DEMO_DIR = find_demo_dir()

# 自由相机：从"机器人背后偏右"看整个工作区（人眼友好的第三人称视角）
FREE_CAM = dict(lookat=(1.60, 0.0, 0.45), distance=1.8, azimuth=150.0, elevation=-18.0)


# ======================================================================
# 文本绘制 / 版式
# ======================================================================
def font(size: int = 14):
    from PIL import ImageFont

    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


BG = (22, 24, 27)
FG = (238, 238, 238)
DIM = (150, 155, 160)
ACCENT = (120, 205, 225)
WARN = (240, 190, 110)


def compose_panel(tiles: Sequence[tuple[str, str, np.ndarray]],
                  title: str, subtitle: str = "", caption: Sequence[str] = (),
                  footer: str = "", cols: int | None = None,
                  pad: int = 8, tile_label_h: int = 20, tile_sub_h: int = 17,
                  caption_line_h: int = 20, header_h: int = 62, footer_h: int = 20):
    """把若干路相机画面拼成一张带注解的版式。

    tiles: [(相机标签, 副标签, 图像(H,W,3) uint8), ...]
    caption: 逐帧变化的说明文字（放在图下方的深色条里）
    """
    from PIL import Image, ImageDraw

    f_title, f_sub, f_tile, f_cap, f_foot = (font(19), font(13), font(12), font(14), font(11))
    n = len(tiles)
    cols = cols or (n if n <= 4 else 3)
    rows = (n + cols - 1) // cols
    th, tw = tiles[0][2].shape[:2]

    cap_h = max(34, caption_line_h * len(caption) + 12) if caption else 0
    W = pad + cols * (tw + pad)
    H = header_h + rows * (tile_label_h + th + tile_sub_h + pad) + cap_h + footer_h
    W += W % 2
    H += H % 2

    panel = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(panel)
    dr.text((pad, 8), title, fill=FG, font=f_title)
    if subtitle:
        dr.text((pad, 33), subtitle, fill=ACCENT, font=f_sub)
    dr.line([(pad, header_h - 7), (W - pad, header_h - 7)], fill=(60, 64, 68))

    y = header_h
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i >= n:
                break
            label, sublabel, img = tiles[i]
            x = pad + c * (tw + pad)
            dr.text((x + 1, y), label, fill=FG, font=f_tile)
            yy = y + tile_label_h
            u8 = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
            panel.paste(Image.fromarray(u8.astype(np.uint8)), (x, yy))
            if sublabel:
                dr.text((x + 1, yy + th + 2), sublabel, fill=DIM, font=f_tile)
        y += tile_label_h + th + tile_sub_h + pad

    if caption:
        dr.rectangle([0, y, W, y + cap_h], fill=(32, 36, 40))
        for k, line in enumerate(caption):
            dr.text((pad, y + 7 + k * caption_line_h), line, fill=FG, font=f_cap)
        y += cap_h
    if footer:
        dr.text((pad, H - footer_h + 3), footer, fill=DIM, font=f_foot)
    return panel


# ======================================================================
# 运动源 1：抓取-放置专家轨迹
# ======================================================================
def _patch_numpy2_gripper() -> bool:
    """给 UranUS 的 env.py 打 numpy>=2 兼容补丁（size-1 数组不能直接 float()）。

    只在内存里 patch，不改他们的仓库文件。已在他们的 env.py 里修过就自动跳过。
    """
    try:
        from src.env import UranusReachEnv
    except Exception:
        return False
    import numpy as _np

    probe = _np.asarray([1.0])
    try:
        float(probe)          # numpy<2 可以；>=2 会 TypeError
        return False
    except TypeError:
        pass

    def _get_gripper(self):   # noqa: ANN001
        return float(np.asarray(self.data.sensor(self.GRIPPER_SENSOR).data).reshape(-1)[0])

    UranusReachEnv.get_gripper = _get_gripper
    return True


def _classify_phase(target: np.ndarray, grip_open: bool, obj0: np.ndarray,
                    place: np.ndarray) -> str:
    """由"目标点位姿 + 夹爪状态"推定当前专家阶段（用于字幕）。

    专家是脚本状态机，每个阶段的笛卡尔目标点都不一样；这里按目标和夹爪开合反推，
    比时间戳切片可靠。区分 approach/lift 靠 grip_open（两者目标点相同只有高度不同）。
    """
    t = np.asarray(target, dtype=float)
    d_obj = float(np.linalg.norm(t - obj0))
    h_obj = float(np.linalg.norm(t[:2] - obj0[:2]))
    h_plc = float(np.linalg.norm(t[:2] - place[:2]))
    if h_plc < 0.03:
        return "搬运到放置点上方 · transport" if t[2] > place[2] + 0.05 else "下放到放置点 · place_down"
    if d_obj < 0.02:
        return "下探到物块 · descend" if grip_open else "闭爪夹持 · close"
    if h_obj < 0.03 and t[2] > obj0[2] + 0.05:
        return "接近物块上方 · approach" if grip_open else "抬升 · lift"
    return "移动 · move"


def collect_expert_trajectory(seed: int = 0, rec_every: int = 25, verbose: bool = True
                              ) -> tuple[list[dict], dict]:
    """跑一次专家抓取-放置，录下每个控制拍的 qpos + 状态量。返回 (snapshots, meta)。"""
    if _DEMO_DIR is None:
        raise RuntimeError(
            "找不到 Lerobot-Uranus-VLA-Demo（UranUS 抓取场景与专家所在仓库）。"
            f"用环境变量 {ENV_VAR} 指定它的路径，或改用 --motion procedural。"
        )
    if str(_DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(_DEMO_DIR))
    patched = _patch_numpy2_gripper()
    if patched and verbose:
        print("  · 已给 src/env.py 的 get_gripper 打 numpy>=2 兼容补丁（仅内存，不改文件）")

    from src.env_grasp import UranusGraspEnv
    from src.expert_grasp import pick_place

    env = UranusGraspEnv(seed=seed)
    rng = np.random.default_rng(seed)
    obj_pose, place_xyz = env.sample_layout(rng)

    # 与 pick_place 内部一致：先 reset / 摆好放置点，才能算准分类用的参考位姿
    env.reset(object_pose=obj_pose)
    if place_xyz is not None:
        env.place_on_table(place_xyz)
    obj0 = env.get_object_pos().copy()
    place = env.get_place_target().copy()

    phase = {"name": "预备位姿 · home"}
    snaps: list[dict] = []
    dt = float(env.model.opt.timestep) * rec_every

    orig_move_to, orig_release = env.move_to, env.release

    def move_to(target, grip_open, **kw):
        phase["name"] = _classify_phase(np.asarray(target, float), bool(grip_open), obj0, place)
        return orig_move_to(target, grip_open, **kw)

    def release(*a, **kw):
        phase["name"] = "缓开爪释放 · release"
        return orig_release(*a, **kw)

    env.move_to, env.release = move_to, release

    def hook():
        snaps.append({
            "qpos": env.data.qpos.copy(),
            "phase": phase["name"],
            "t": len(snaps) * dt,
            "obj_z": float(env.get_object_pos()[2]),
            "gap": float(env.tooth_gap()),
            "dist": float(np.linalg.norm(env.tooth_midpoint() - env.get_object_pos())),
            "grip": float(env.get_gripper()),
        })

    res = pick_place(env, obj_pose=obj_pose, place_xyz=place_xyz,
                     record=True, rec_every=rec_every, frame_hook=hook)

    meta = {"source": "expert", "ok": bool(res["ok"]), "lift": float(res["lift"]),
            "horiz_err": float(res["horiz_err"]), "steps": int(res["steps"]),
            "obj0": np.asarray(res["obj0"]).tolist(), "place": np.asarray(res["place"]).tolist(),
            "rec_every": rec_every, "dt": dt, "seed": seed,
            "model": env.model, "obj0_z": float(obj0[2])}
    if verbose:
        print(f"  · 专家轨迹：{res['steps']} 物理步 / {len(snaps)} 个控制拍，"
              f"抬升 {res['lift'] * 1000:.0f}mm，成功={res['ok']}")
    return snaps, meta


def collect_procedural_trajectory(n_frames: int = 140, verbose: bool = True
                                  ) -> tuple[list[dict], dict]:
    """兜底：不依赖 UranUS 环境，直接用 keyframe 附近的关节小幅摆动。"""
    import mujoco

    from .demo import load_scene

    model, data, desc = load_scene(None)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    q0 = data.qpos.copy()
    snaps = []
    for i in range(n_frames):
        u = i / max(n_frames - 1, 1)
        q = q0.copy()
        q[7] += 0.25 * np.sin(2 * np.pi * u)            # joint2
        q[8] += 0.20 * np.sin(2 * np.pi * u + 0.6)      # joint3
        q[9] += 0.15 * np.sin(4 * np.pi * u)            # joint4
        snaps.append({"qpos": q, "phase": "程序化摆动（无专家）", "t": i * 0.05,
                      "obj_z": float(q[2]), "gap": 0.0, "dist": 0.0, "grip": float(q[13])})
    if verbose:
        print(f"  ！专家不可用，退回程序化运动（{n_frames} 帧，语义弱）")
    return snaps, {"source": "procedural", "ok": False, "lift": 0.0, "horiz_err": 0.0,
                   "steps": n_frames, "model": model, "obj0_z": float(q0[2]),
                   "scene": desc, "rec_every": 1, "dt": 0.05, "seed": 0}


# ======================================================================
# 渲染 / 拼版
# ======================================================================
def free_camera(lookat=(0.0, 0.0, 0.0), distance: float = 2.0,
                azimuth: float = 90.0, elevation: float = -20.0):
    """构造一个自由相机（MjvCamera），可以当作 ``camera=`` 传给 WaterRenderer。

    自由相机用的是 ``model.vis.global_.fovy``（默认 45°），WaterRenderer 会自动取。
    """
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def luma_contrast(img: np.ndarray) -> float:
    a = np.asarray(img, np.float32)
    a = a / 255.0 if img.dtype == np.uint8 else a
    return float((a @ np.array([0.299, 0.587, 0.114], np.float32)).std())


def make_snap_data(model, qpos: np.ndarray):
    """把录下来的 qpos 变成一个可以渲染的 MjData（只做正运动学，不跑动力学）。"""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data


def render_tiles(wr: WaterRenderer, model, snap: dict,
                 cells: Sequence[dict]) -> list[tuple[str, str, np.ndarray]]:
    """一帧：依次渲染各个 cell（每个 cell 可带自己的水况/光照/机位）。"""
    import mujoco

    data = make_snap_data(model, snap["qpos"])
    out: list[tuple[str, str, np.ndarray]] = []
    imgs: list[np.ndarray] = []
    subs: list[str] = []
    for cell in cells:
        if cell.get("params") is not None:
            wr.set_params(cell["params"])
        if cell.get("style") is not None:
            wr.set_style(cell["style"])
        imgs.append(wr.render(data, camera=cell["camera"]))
        subs.append(cell.get("sub", ""))
    del data, mujoco

    # 相对对比度（以本帧第一个 cell 为基准）——消融图的关键数字
    base = max(luma_contrast(imgs[0]), 1e-6)
    for cell, img, sub in zip(cells, imgs, subs):
        extra = cell.get("sub_auto")
        if extra:
            sub = (sub + "  " if sub else "") + extra.format(contrast=luma_contrast(img) / base)
        out.append((cell["label"], sub, img))
    return out


# ======================================================================
# 三张动画
# ======================================================================
def anim_defs() -> dict[str, dict]:
    """三张动画的配置：cells = 每张图里并排的几路画面。"""
    ov = free_camera(**FREE_CAM)
    # task / water 两张图里光照**不是**自变量，取可读性最好的表层光照
    # （实测 coastal + deep 会把腕部相机对比度压到 0.03，几乎没细节；
    #   surface 下 grasp 0.155 / wrist 0.100 / overview 0.132，三路都看得清）
    style_base = SceneStyle.preset("surface")
    return {
        "task": dict(
            title="UranUS 水下机械臂 · 抓取-放置（三视角）",
            subtitle="MuJoCo 3.8 离屏渲染 + underwater_vision 水下成像模组",
            water_line=lambda meta: (f"水况 coastal（β_R={WaterParams.preset('coastal').beta[0]:.2f}/m，"
                                     f"能见度≈{WaterParams.preset('coastal').visibility_m:.1f}m）"
                                     f"   ·   光照 surface   ·   深度=逐像素   ·   轨迹：脚本专家状态机"),
            cells=[
                dict(label="① 场景相机 grasp_cam（俯视侧视 · fovy 50°）", camera="grasp_cam",
                     params="coastal", style=style_base, sub="策略输入视角", sub_auto="相对对比度 {contrast:.2f}×"),
                dict(label="② 腕部相机 wrist_cam（贴夹爪 · fovy 60°）", camera="wrist_cam",
                     params="coastal", style=style_base, sub="近距 → 背向散射更重",
                     sub_auto="相对对比度 {contrast:.2f}×"),
                dict(label="③ 第三人称 overview（自由相机 · fovy 45°）", camera=ov,
                     params="coastal", style=style_base, sub="人眼友好视角",
                     sub_auto="相对对比度 {contrast:.2f}×"),
            ],
            cols=3,
        ),
        "water": dict(
            title="同一段动作 · 不同水况（场景相机）",
            subtitle="水况是唯一的自变量：几何、光照、轨迹全部相同；水体光学由 underwater_vision 逐像素计算",
            water_line=lambda meta: "clean → coastal → turbid → harbor：能见度与背向散射逐级恶化",
            cells=[
                dict(label="① clean（清水基准）", camera="grasp_cam", params="clear", style=style_base,
                     sub="不加水体成像，仅作对照", sub_auto="基准 {contrast:.2f}×"),
                dict(label="② coastal 沿岸", camera="grasp_cam", params="coastal", style=style_base,
                     sub="能见度≈4.3m", sub_auto="{contrast:.2f}×"),
                dict(label="③ turbid 浑浊", camera="grasp_cam", params="turbid", style=style_base,
                     sub="能见度≈2.1m，红通道近乎消失", sub_auto="{contrast:.2f}×"),
                dict(label="④ harbor 港口泥沙", camera="grasp_cam", params="harbor", style=style_base,
                     sub="能见度≈1.2m", sub_auto="{contrast:.2f}×"),
            ],
            cols=4,
        ),
        "light": dict(
            title="同一段动作 · 不同水下光照（沿岸水况）",
            subtitle="水况固定为 coastal，只改探照灯/环境光：越深越依赖探照灯，代价是热点与暗区",
            water_line=lambda meta: "surface → mid → deep → dark：headlight 强度与环境光逐级下降",
            cells=[
                dict(label="① surface 表层", camera="grasp_cam", params="coastal",
                     style=SceneStyle.preset("surface"), sub="环境光为主", sub_auto="相对对比度 {contrast:.2f}×"),
                dict(label="② mid 中层", camera="grasp_cam", params="coastal",
                     style=SceneStyle.preset("mid"), sub="环境光衰减，探照灯接管",
                     sub_auto="{contrast:.2f}×"),
                dict(label="③ deep 深水", camera="grasp_cam", params="coastal",
                     style=SceneStyle.preset("deep"), sub="几乎全黑 + 强探照灯",
                     sub_auto="{contrast:.2f}×"),
                dict(label="④ dark 关灯巡检", camera="grasp_cam", params="coastal",
                     style=SceneStyle.preset("dark"), sub="只剩极弱环境光", sub_auto="{contrast:.2f}×"),
            ],
            cols=4,
        ),
    }


def build(name: str, snaps: list[dict], meta: dict, out_dir: Path, size: int = 200,
          fps: float = 10.0, formats: Iterable[str] = ("gif", "mp4"),
          gif_colors: int = 128, verbose: bool = True) -> dict[str, Path]:
    """渲染一张动画并保存。返回 {格式: 路径}。"""
    cfg = anim_defs()[name]
    model = meta["model"]
    wr = WaterRenderer(model, size, size, params=cfg["cells"][0]["params"] or "clear",
                       seed=0, use_depth=True)

    def _ph(p) -> str:
        if isinstance(p, WaterParams):
            return p.hash()
        return WaterParams.preset(p).hash() if isinstance(p, str) else "-"

    group_hash = hashlib.sha1("|".join(_ph(c.get("params")) for c in cfg["cells"]).encode()
                              ).hexdigest()[:8]
    footer = (f"{cfg['water_line'](meta)}   ·   underwater_vision v{_version()}"
              f"   ·   参数组 hash {group_hash}")

    t0 = time.time()
    frames = []
    for i, snap in enumerate(snaps):
        tiles = render_tiles(wr, model, snap, cfg["cells"])
        lift_mm = (snap["obj_z"] - meta["obj0_z"]) * 1000.0
        caption = [
            f"t = {snap['t']:.2f} s    帧 {i + 1}/{len(snaps)}",
            f"阶段：{snap['phase']}",
            (f"物块抬升 {lift_mm:+7.1f} mm    齿面-物块间隙 {snap['gap'] * 1000:5.1f} mm    "
             f"夹爪 {snap['grip']:.3f} rad"),
        ]
        cap = caption if meta["source"] == "expert" else caption[:2]
        panel = compose_panel(tiles, cfg["title"], cfg["subtitle"], cap,
                              footer=footer, cols=cfg["cols"])
        frames.append(panel)
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(snaps)} 帧  ({time.time() - t0:.0f}s)")
    wr.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    # 顺带存一张静态图（报告里当插图/缩略图用，注解与动画完全一致）
    still = out_dir / f"{name}_still.png"
    frames[len(frames) // 2].save(still)
    paths["still"] = still
    if "gif" in formats:
        p = out_dir / f"{name}.gif"
        save_gif(frames, p, fps, colors=gif_colors)
        paths["gif"] = p
    if "mp4" in formats:
        p = out_dir / f"{name}.mp4"
        if save_mp4(frames, p, fps):
            paths["mp4"] = p
        elif verbose:
            print("    ！MP4 写入失败（cv2 编解码器不可用），只出 GIF")
    if verbose:
        print(f"  · {name}: {len(frames)} 帧，{time.time() - t0:.0f}s -> "
              + ", ".join(f"{k}={v}" for k, v in paths.items()))
    return paths


def save_gif(frames, path: Path, fps: float, colors: int = 128) -> Path:
    """存 GIF。``colors`` 越小文件越小（128 一般够看；报告主推 MP4，色彩无损）。"""
    from PIL import Image

    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=int(colors)) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(round(1000.0 / fps)),
                loop=0, optimize=True, disposal=2)
    return path


def save_mp4(frames, path: Path, fps: float) -> bool:
    try:
        import cv2
    except Exception:
        return False
    h, w = frames[0].size[1], frames[0].size[0]
    try:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not vw.isOpened():
            return False
        for f in frames:
            vw.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
        vw.release()
    except Exception:
        return False
    return path.exists() and path.stat().st_size > 0


def _version() -> str:
    from . import __version__

    return __version__


# ======================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="报告用动画生成器（多视角 + 中文注解）")
    ap.add_argument("--anim", type=str, default=None, choices=["task", "water", "light"])
    ap.add_argument("--all", action="store_true", help="三张都出")
    ap.add_argument("--list", action="store_true", help="列出可用动画")
    ap.add_argument("--out", type=str, default=str(_HERE / "out" / "anim"))
    ap.add_argument("--size", type=int, default=200, help="单个相机画面的边长(px)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=2, help="每 N 个控制拍取一帧（默认 2 → 10fps）")
    ap.add_argument("--max-frames", type=int, default=72)
    ap.add_argument("--rec-every", type=int, default=25, help="每多少个物理步录一拍（25*2ms=50ms）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formats", type=str, default="gif,mp4")
    ap.add_argument("--gif-colors", type=int, default=128,
                    help="GIF 调色板颜色数（越小文件越小；报告建议直接用 MP4）")
    ap.add_argument("--motion", type=str, default="auto", choices=["auto", "expert", "procedural"])
    args = ap.parse_args(argv)

    if args.list or not (args.anim or args.all):
        print("可用动画：")
        for k, v in anim_defs().items():
            print(f"  {k:6s} {v['title']}   （{len(v['cells'])} 路画面，cols={v['cols']}）")
        if not args.list:
            print("\n用 --anim <名字> 或 --all 生成。")
        return 0

    names = ["task", "water", "light"] if args.all else [args.anim]
    print(f"运动源：{args.motion}  帧上限 {args.max_frames}  stride {args.stride}")

    if args.motion == "procedural":
        snaps, meta = collect_procedural_trajectory()
    else:
        try:
            snaps, meta = collect_expert_trajectory(seed=args.seed, rec_every=args.rec_every)
        except Exception as e:
            if args.motion == "expert":
                raise
            print(f"  ！专家轨迹不可用（{type(e).__name__}: {e}）")
            snaps, meta = collect_procedural_trajectory()

    snaps = snaps[::args.stride]
    if len(snaps) > args.max_frames:
        idx = np.linspace(0, len(snaps) - 1, args.max_frames).astype(int)
        snaps = [snaps[i] for i in idx]
    print(f"  · 用于动画：{len(snaps)} 帧（{len(snaps) / args.fps:.1f}s @ {args.fps}fps）")

    fmt = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    for name in names:
        build(name, snaps, meta, Path(args.out), size=args.size, fps=args.fps, formats=fmt,
              gif_colors=args.gif_colors)
    print(f"\n输出目录：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())