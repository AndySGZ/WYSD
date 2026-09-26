# -*- coding: utf-8 -*-
"""障碍抓取族的**拍摄展示**：把每级场景渲染成图片（不接水下模组，纯 MuJoCo）。

    python -m obstacle_grasp.shots                 # 每级卡片 + 五级总览
    python -m obstacle_grasp.shots --tile 420
    python -m obstacle_grasp.shots --levels o3_front_overhang

产出（在 `shots/`）：
  - `overview.png`     五级总览拼版（策略视角 × 5 + 名称/障碍/难度轴）—— 一张图讲清整条阶梯
  - `<n>_<level>.png`  每级一张卡片：两路渲染（策略视角 + 第三人称）+ 该级规格信息
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .env import ObstacleGraspEnv
from .tasks import LEVEL_ORDER, get

_HERE = Path(__file__).resolve().parent

BG = (22, 24, 28)
FG = (240, 240, 240)
DIM = (150, 156, 164)
ACCENT = (240, 190, 100)
LABEL = (170, 205, 235)
RULE = (60, 64, 72)

# 第三人称自由相机（人眼友好）：与 underwater_vision.animate 用的是同一组参数
FREE_CAM = dict(lookat=(1.60, 0.0, 0.45), distance=1.8, azimuth=150.0, elevation=-18.0)


def _font(size: int = 15):
    from PIL import ImageFont

    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap(draw, text: str, font, max_w: int) -> list[str]:
    """中文按字断行。"""
    lines, cur = [], ""
    for ch in str(text):
        trial = cur + ch
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _render(level: str, tile: int, no_text: bool = False):
    """渲染两路视角，返回 [(标签, ndarray), ...]。纯 mujoco.Renderer，不做水下处理。"""
    import mujoco

    env = ObstacleGraspEnv(level, seed=0)
    env.reset()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = FREE_CAM["lookat"]
    cam.distance = FREE_CAM["distance"]
    cam.azimuth = FREE_CAM["azimuth"]
    cam.elevation = FREE_CAM["elevation"]

    r = mujoco.Renderer(env.model, tile, tile)
    out = []
    for cam_arg in ("grasp_cam", cam):
        r.update_scene(env.data, camera=cam_arg)
        out.append(r.render().copy())
    r.close()
    return out, env


def _rows(env: ObstacleGraspEnv) -> list[tuple[str, str]]:
    s = env.spec
    instr = " ｜ ".join(f"{i + 1}. {t}" for i, t in enumerate(s["instructions"]))
    return [
        ("障碍物", s["obstacle"]),
        ("新增难度轴", s["difficulty"]),
        ("成功判据", s["criterion"] + f"（需连续成立 {s['hold_s']:.1f}s）"),
        ("场景文件", f"scenes/{Path(s['xml']).name}"),
        ("语言指令池", instr),
        ("hold-out", s["holdout"][0] + "（不参与训练，只评测）"),
        ("实现要点", s["notes"]),
    ]


def build_card(level: str, tile: int = 380, out_dir: Path | None = None) -> Path:
    from PIL import Image, ImageDraw

    imgs, env = _render(level, tile)
    s = env.spec
    f_title, f_sub, f_tile, f_lbl, f_val, f_foot = (_font(28), _font(15), _font(14),
                                                    _font(15), _font(15), _font(12))
    pad = 16
    W = pad + 2 * (tile + pad)
    info_w = W - 2 * pad

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = [(lab, wrap(probe, val, f_val, info_w - 118)) for lab, val in _rows(env)]

    header = 104
    tiles_h = 40 + tile + 18 + pad
    info_h = sum(20 * max(1, len(ls)) + 8 for _, ls in wrapped) + 16
    foot = 34
    H = header + tiles_h + info_h + foot
    W += W % 2
    H += H % 2

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 14), f"O{s['order']}  {s['title'].split(' ', 1)[-1]}", fill=FG, font=f_title)
    fam = "族 O · 障碍抓取（不做水下渲染）"
    tag = f"{fam}    等级 {s['order']}/{len(LEVEL_ORDER)}"
    d.text((W - pad - d.textlength(tag, font=f_sub), 22), tag, fill=ACCENT, font=f_sub)
    d.text((pad, 52), f"新增难度轴：{s['difficulty']}", fill=LABEL, font=f_sub)
    d.text((pad, 76), "与 underwater_sampling（水下族）正交：那一族考「看不清」，这一族考「看不见 / 够不着 / 路被堵」",
           fill=DIM, font=f_foot)

    y = header
    labels = ("① 策略输入视角  grasp_cam  fovy 50°", "② 第三人称视角（自由相机）")
    for i, arr in enumerate(imgs):
        x = pad + i * (tile + pad)
        d.text((x + 2, y), labels[i], fill=FG, font=f_tile)
        img.paste(Image.fromarray(arr), (x, y + 22))
    y += 22 + tile + 18
    d.line([(pad, y - 9), (W - pad, y - 9)], fill=RULE)

    for lab, lines in wrapped:
        first = True
        for ln in lines:
            d.text((pad, y), lab if first else "", fill=LABEL, font=f_lbl)
            d.text((pad + 110, y), ln, fill=FG, font=f_val)
            first = False
            y += 20
        y += 8

    d.text((pad, H - foot + 6),
           f"obstacle_grasp    规格 hash {env.info()['spec_hash']}    "
           f"几何约束为实测值（O3 低梁默认姿态咬入 12.3mm；O4 窄缝每侧余量 4.9mm）",
           fill=DIM, font=f_foot)

    out_dir = out_dir or (_HERE / "shots")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{s['order']}_{level}.png"
    img.save(path)
    return path


def build_overview(tile: int = 300, out_dir: Path | None = None) -> Path:
    """五级总览：策略视角并排 + 名称/障碍/难度轴 —— 一张图讲清整条阶梯。"""
    from PIL import Image, ImageDraw

    f_title, f_sub, f_name, f_body, f_foot = (_font(30), _font(14), _font(19),
                                              _font(13), _font(12))
    items = []
    for lv in LEVEL_ORDER:
        imgs, env = _render(lv, tile)
        items.append((env, imgs[0]))

    pad, gap = 18, 12
    n = len(items)
    W = pad + n * (tile + gap)
    text_h = 150
    header, foot = 96, 40
    H = header + tile + 34 + text_h + foot
    W += W % 2
    H += H % 2

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 16), "障碍抓取族 · 五级阶梯总览", fill=FG, font=f_title)
    d.text((pad, 56), "O1 无障碍基线 → 视觉遮挡 → 路径阻挡 → 毫米级精度 → 长程顺序；"
                      "每级只加一个难度轴，O1 与其它级的成功率差就是「障碍的代价」",
           fill=DIM, font=f_sub)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    y = header
    for i, (env, arr) in enumerate(items):
        x = pad + i * (tile + gap)
        d.rectangle([x - 2, y - 2, x + tile + 2, y + tile + 32], fill=(30, 33, 38))
        d.text((x + 2, y + 2), f"O{env.spec['order']}  {env.spec['title'].split(' ', 1)[-1]}",
               fill=ACCENT, font=f_name)
        img.paste(Image.fromarray(arr), (x, y + 28))
        yy = y + 28 + tile + 8
        for lab, txt in (("障碍", env.spec["obstacle"]),
                         ("难度轴", env.spec["difficulty"])):
            for k, ln in enumerate(wrap(probe, f"{lab}：{txt}", f_body, tile - 4)):
                if yy > y + tile + 30:
                    break
                d.text((x + 2, yy), ln, fill=FG if k == 0 else DIM, font=f_body)
                yy += 17
    d.text((pad, H - foot + 8),
           "每张卡片（含两路渲染 + 完整规格）见同目录 <n>_<level>.png    "
           "python -m obstacle_grasp.shots 可重跑", fill=DIM, font=f_foot)

    out_dir = out_dir or (_HERE / "shots")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overview.png"
    img.save(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="障碍抓取族：场景拍摄展示")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--tile", type=int, default=380)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else None
    levels = args.levels or LEVEL_ORDER
    print(f"渲染 {len(levels)} 级场景卡片 -> {out or (_HERE / 'shots')}")
    for lv in levels:
        get(lv)
        p = build_card(lv, tile=args.tile, out_dir=out)
        print(f"  · {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    if not args.no_overview and not args.levels:
        p = build_overview(tile=max(240, args.tile - 80), out_dir=out)
        print(f"  · {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
