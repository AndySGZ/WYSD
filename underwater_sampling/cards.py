# -*- coding: utf-8 -*-
"""把每一级场景连介绍信息打成一张卡片图（报告/评审/贴文档用）。

    python -m underwater_sampling.cards                    # 七张 -> cards/
    python -m underwater_sampling.cards --levels l1_rock_collect l5_push_core
    python -m underwater_sampling.cards --tile 380 --preset clear --no-depth

每张卡片 = 两路真实渲染（① 策略输入视角 grasp_cam ② 第三人称俯视自由相机）
+ 该级的规格信息（难度轴 / 被操作物 / 目标 / 成功判据 / 保持时长 / 建议水况 /
场景文件 / 语言指令池 + hold-out / 实现要点）。渲染走 `underwater_vision`，所以卡片上的
画面就是策略将来看到的水下图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .env import SamplingEnv
from .tasks import LEVEL_ORDER, get

_HERE = Path(__file__).resolve().parent

BG = (20, 22, 26)
PANEL = (30, 34, 40)
FG = (238, 238, 238)
DIM = (152, 158, 165)
ACCENT = (122, 206, 226)
LABEL = (168, 214, 236)
RULE = (58, 63, 70)

# 族标签（tasks.py 里每个等级都有 family 字段；L1~L7 = A 采样，L8 = B 干预）
FAMILY = {"A": "族 A · 科学/生物采样", "B": "族 B · 工业干预"}


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
    """中文按字断行（没有空格可依），英文/数字尽量不切断单词。"""
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


def _views(level: str, tile: int, preset: str | None, use_depth: bool):
    """渲染两路视角，返回 [(标签, 副标签, ndarray), ...]。"""
    import mujoco

    from underwater_vision import WaterRenderer
    from underwater_vision.animate import FREE_CAM, free_camera

    env = SamplingEnv(level, seed=0)
    env.reset()
    p = preset or env.spec["water"]
    wr = WaterRenderer(env.model, tile, tile, params=p, seed=0, use_depth=use_depth)
    cam_ov = free_camera(**FREE_CAM)
    fovy = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i):
            float(env.model.cam_fovy[i]) for i in range(env.model.ncam)}
    out = [
        ("① 策略输入视角  grasp_cam", f"fovy {fovy.get('grasp_cam', 50):.0f}°   水况 {p}",
         wr.render(env.data, camera="grasp_cam")),
        ("② 第三人称视角（自由相机）", f"fovy {float(env.model.vis.global_.fovy):.0f}°   人眼友好",
         wr.render(env.data, camera=cam_ov)),
    ]
    wr.close()
    return out, float(env.model.vis.global_.fovy)


def _rows(env: SamplingEnv) -> list[tuple[str, str]]:
    s = env.spec
    instr = " ｜ ".join(f"{i + 1}. {t}" for i, t in enumerate(s["instructions"]))
    rowlist = [
        ("新增难度轴", s["difficulty"]),
        ("被操作物", s["object_desc"]),
        ("目标", s["goal_desc"]),
        ("成功判据", s.get("criterion", "") + f"（判据需连续成立 {s['hold_s']:.1f}s）"),
        ("建议水况", f"{s['water']}（underwater_vision 预设；采集与评测必须用同一组参数）"),
        ("场景文件", f"scenes/{Path(s['xml']).name}"),
        ("语言指令池", instr),
        ("hold-out", s["holdout"][0] + "（不参与训练，只评测）"),
        ("实现要点", s["notes"]),
    ]
    if "tol" in s:
        rowlist.insert(4, ("判据阈值", "  ".join(f"{k}={v}" for k, v in s["tol"].items())))
    return rowlist


def build_card(level: str, tile: int = 340, preset: str | None = None,
               use_depth: bool = True, out_dir: Path | None = None) -> Path:
    from PIL import Image, ImageDraw

    env = SamplingEnv(level, seed=0)
    env.reset()
    s = env.spec
    views, _ = _views(level, tile, preset, use_depth)

    f_title, f_sub, f_tile, f_lbl, f_val, f_foot = (_font(27), _font(15), _font(14),
                                                    _font(15), _font(15), _font(12))
    pad = 16
    W = pad + 2 * (tile + pad)
    info_w = W - 2 * pad

    # 先用一张临时图量文字行数，再定最终高度
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped: list[tuple[str, list[str]]] = []
    for label, value in _rows(env):
        wrapped.append((label, wrap(probe, value, f_val, info_w - 118)))

    header = 106
    tiles_h = 24 + tile + 22 + pad
    info_h = sum(20 * max(1, len(ls)) + 8 for _, ls in wrapped) + 16
    foot = 34
    H = header + tiles_h + info_h + foot
    W += W % 2
    H += H % 2

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 16), f"L{s['order']}  {s['title'].split(' ', 1)[-1]}", fill=FG, font=f_title)
    fam = FAMILY.get(s.get("family", "A"), f"族 {s.get('family', '?')}")
    tag = f"{fam}    等级 {s['order']}/{len(LEVEL_ORDER)}"
    d.text((W - pad - d.textlength(tag, font=f_sub), 24), tag, fill=ACCENT, font=f_sub)
    d.text((pad, 52), f"新增难度轴：{s['difficulty']}", fill=LABEL, font=f_sub)
    d.text((pad, 76), f"水下采样任务族 · 数据由人手操作采集 · 无专家策略", fill=DIM, font=f_foot)

    y = header
    for i, (label, sub, arr) in enumerate(views):
        x = pad + i * (tile + pad)
        d.text((x + 2, y), label, fill=FG, font=f_tile)
        d.text((x + 2, y + 18), sub, fill=DIM, font=f_foot)
        img.paste(Image.fromarray(arr), (x, y + 40))
        y_end = y + 40 + tile
    y = y_end + 22
    d.line([(pad, y - 10), (W - pad, y - 10)], fill=RULE)

    for label, lines in wrapped:
        first = True
        for ln in lines:
            d.text((pad, y), label if first else "", fill=LABEL, font=f_lbl)
            d.text((pad + 110, y), ln, fill=FG, font=f_val)
            first = False
            y += 20
        y += 8

    foot_txt = (f"underwater_sampling v{_version()}   规格 hash {env.info()['spec_hash']}   "
                f"判据由 selftest 场景自检保证（IK 可达 / 摆位咬入 <2mm / 物理静置）")
    d.text((pad, H - foot + 6), foot_txt, fill=DIM, font=f_foot)

    out_dir = out_dir or (_HERE / "cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{s['order']}_{level}.png"
    img.save(path)
    env.close() if hasattr(env, "close") else None
    return path


def _version() -> str:
    from . import __version__

    return __version__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把每级场景打成一张带介绍信息的卡片图")
    ap.add_argument("--levels", nargs="*", default=None, help="默认全部七级")
    ap.add_argument("--tile", type=int, default=340, help="单路视角的边长(px)")
    ap.add_argument("--preset", type=str, default=None,
                    help="覆盖水况预设（默认用该级的建议水况）")
    ap.add_argument("--no-depth", dest="use_depth", action="store_false", default=True,
                    help="不做逐像素深度（更快，但衰减不随距离变化）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    levels = args.levels or LEVEL_ORDER
    out = Path(args.out) if args.out else None
    print(f"生成 {len(levels)} 张场景卡片 -> {out or (_HERE / 'cards')}")
    for lv in levels:
        get(lv)                                   # 校验等级名
        p = build_card(lv, tile=args.tile, preset=args.preset,
                       use_depth=args.use_depth, out_dir=out)
        print(f"  · {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
