# -*- coding: utf-8 -*-
"""在终端里"看图"：把图像打成 ASCII 亮度图 + 统计。

用途：无 GUI / 无法直接查看图片时，快速确认渲染结果的结构与退化程度。
支持 PNG 等图片格式（需要 PIL）与 ``.npy``。

    python -m underwater_vision.preview out/grasp_cam_clean_*.png out/grasp_cam_turbid.png
    python -m underwater_vision.preview out/grasp_cam_sheet.png --cols 100
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

RAMP = " .:-=+*#%@"

__all__ = ["ascii_map", "load_image", "stats", "preview"]


def load_image(path: str | Path) -> np.ndarray:
    """读图片（PIL）或 .npy。"""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    from PIL import Image

    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def _to_float(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def stats(img: np.ndarray) -> dict:
    a = _to_float(img)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return {
        "shape": tuple(np.asarray(img).shape),
        "mean": float(lum.mean()),
        "rgb": [round(float(v), 3) for v in a.reshape(-1, 3).mean(0)],
        "contrast": float(lum.std()),
        "dark_pct": float((lum < 0.16).mean() * 100.0),
    }


def ascii_map(img: np.ndarray, cols: int = 64, rows: int = 24, ramp: str = RAMP) -> str:
    """把图像降采样成 ASCII 亮度图（宽高比按字符单元 2:1 校正）。"""
    a = _to_float(img)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    if lum.ndim != 2:
        raise ValueError(f"需要二维亮度图，收到 {lum.shape}")
    h, w = lum.shape
    rows = max(1, min(rows, h))
    cols = max(1, min(cols, w * 2))     # 字符是"高瘦"的，列数一般给宽度的 2 倍
    yi = np.linspace(0, h - 1, rows).astype(int)
    xi = np.linspace(0, w - 1, cols).astype(int)
    small = lum[yi][:, xi]
    idx = np.clip((small * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    return "\n".join("".join(ramp[i] for i in row) for row in idx)


def preview(paths, cols: int = 64, rows: int = 24, show_stats: bool = True,
            ramp: str = RAMP) -> None:
    for path in paths:
        try:
            img = load_image(path)
        except Exception as e:
            print(f"！读不了 {path}：{e}")
            continue
        print(f"\n=== {path} ===")
        if show_stats:
            st = stats(img)
            print(f"  shape={st['shape']}  mean={st['mean'] * 255:.1f}  "
                  f"RGB={[round(v * 255, 1) for v in st['rgb']]}  "
                  f"contrast={st['contrast']:.3f}  dark={st['dark_pct']:.1f}%")
        print(ascii_map(img, cols=cols, rows=rows, ramp=ramp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="终端看图（ASCII）")
    ap.add_argument("paths", nargs="*", help="图片路径 / glob，如 out/*.png")
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--ramp", type=str, default=RAMP)
    args = ap.parse_args(argv)

    paths: list[str] = []
    for p in args.paths:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else [p])
    if not paths:
        ap.print_help()
        return 2

    preview(paths, cols=args.cols, rows=args.rows, ramp=args.ramp)
    return 0


if __name__ == "__main__":
    sys.exit(main())