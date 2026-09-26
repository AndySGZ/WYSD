# -*- coding: utf-8 -*-
"""自检 demo：在真实 UranUS 抓取场景上渲染"清水 vs 水下"对照图。

    python -m underwater_vision.demo                 # 用默认 UranUS 场景
    python -m underwater_vision.demo --xml path.xml  # 换场景
    python -m underwater_vision.demo --list-presets  # 只看预设

输出（默认写到 ./out/）：
    <cam>_<标签>.png        每一张单独的图（清水图也在，方便对照）
    <cam>_sheet.png         拼版对照图（含每组的统计）
    demo_report.json        WaterRenderer.info() —— 直接抄进 dataset meta

不做任何"假装在渲染器里做雾"的事：渲染器只出清水图，水下图全部由
``apply_water`` 在图像域生成（原因见包 docstring）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .params import PRESETS, WaterParams, preset_names
from .paths import scene_candidates
from .render import WaterRenderer, save_png

_HERE = Path(__file__).resolve().parent

# 场景候选由 paths.py 向上逐层探测（兼容本模块放在仓库外 / 仓库根内两种位置）
_SCENE_CANDIDATES = scene_candidates()

# 找不到任何场景时的兜底：一个自带的小场景（保证 demo 在任何环境下都能跑）
_BUILTIN_XML = """
<mujoco model="underwater_vision_demo">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.55 0.55 0.55" specular="0.10 0.10 0.10"/>
    <quality shadowsize="1024"/>
    <map znear="0.02" zfar="30"/>
  </visual>
  <worldbody>
    <light pos="1.0 1.0 2.0" dir="-1 -1 -2" diffuse="0.70 0.70 0.70"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.05" rgba="0.55 0.52 0.45 1"/>
    <body name="cube" pos="0.40 0.10 0.045">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.045 0.045 0.045" rgba="0.95 0.75 0.20 1" friction="0.6 0.02 0.001"/>
    </body>
    <body name="pipe" pos="0.42 -0.18 0.06">
      <geom name="pipe_geom" type="cylinder" size="0.035 0.30" quat="0.7071 0 0.7071 0" rgba="0.50 0.52 0.50 1"/>
    </body>
    <body name="plate" pos="0.36 0.28 0.01">
      <geom name="plate_geom" type="cylinder" size="0.06 0.006" rgba="0.20 0.45 0.85 1"/>
    </body>
    <camera name="grasp_cam" pos="1.20 -1.00 0.90" xyaxes="1 0 0 0 0.5 0.866" fovy="50"/>
    <camera name="wrist_cam" pos="0.42 -0.05 0.35" xyaxes="1 0 0 0 0.5 0.866" fovy="60"/>
  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------
def load_scene(xml: str | None):
    """返回 (model, data, 场景描述)。"""
    import mujoco

    if xml:
        path = Path(xml)
        model = mujoco.MjModel.from_xml_path(str(path))
        desc = str(path)
    else:
        found = next((p for p in _SCENE_CANDIDATES if p.exists()), None)
        if found is not None:
            model = mujoco.MjModel.from_xml_path(str(found))
            desc = str(found)
        else:
            model = mujoco.MjModel.from_xml_string(_BUILTIN_XML)
            desc = "<内置兜底场景>"

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, desc


def img_stats(img: np.ndarray) -> dict[str, float]:
    a = np.asarray(img, dtype=np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    luma = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return {
        "mean": float(a.mean()),
        "rgb": [round(float(v), 3) for v in a.reshape(-1, 3).mean(axis=0)],
        "contrast": float(luma.std()),
        "dark_pct": float((luma < 0.16).mean() * 100.0),
    }


def _font(size: int = 14):
    from PIL import ImageFont

    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def compose_sheet(rows, out_path: Path, title: str) -> Path:
    """rows: [(组标题, [(图标签, 副标签, 图像), ...]), ...]"""
    from PIL import Image, ImageDraw

    f_title, f_row, f_tile = _font(17), _font(14), _font(11)
    pad, label_h, sub_h = 8, 20, 15
    ncol = max(len(r[1]) for r in rows)
    th, tw = rows[0][1][0][2].shape[:2]
    W = pad + ncol * (tw + pad)
    H = pad + 34 + len(rows) * (label_h + th + sub_h + pad) + 10

    sheet = Image.new("RGB", (W, H), (24, 26, 28))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 8), title, fill=(235, 235, 235), font=f_title)
    y = pad + 34
    for row_label, tiles in rows:
        dr.text((pad, y), row_label, fill=(150, 200, 220), font=f_row)
        y += label_h
        x = pad
        for tlabel, sublabel, img in tiles:
            u8 = np.clip(img, 0, 1) * 255 if img.dtype != np.uint8 else img
            sheet.paste(Image.fromarray(u8.astype(np.uint8)), (x, y))
            dr.text((x + 2, y + th + 1), tlabel, fill=(225, 225, 225), font=f_tile)
            if sublabel:
                dr.text((x + 2, y + th + 1 + 13), sublabel, fill=(150, 150, 150), font=f_tile)
            x += tw + pad
        y += th + sub_h + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ----------------------------------------------------------------------
def build_groups(wr: WaterRenderer, data, cam: str, seed: int, base_style: str):
    """四组对照：预设 / 能见度扫描 / 域随机化 / 光照风格。"""
    rng = np.random.default_rng(seed + 12345)
    rows = []

    # --- 1. 清水基准 + 各预设 ---
    wr.set_style(base_style)
    clean = wr.clean(data, cam)
    tiles = [("clean (no water)", "", clean)]
    for name in ("clear", "coastal", "turbid", "deep", "harbor"):
        wr.set_params(name)
        tiles.append((name, f"vis {wr.params.visibility_m:.1f}m", wr.render(data, cam)))
    rows.append(("presets  —  clean reference then water presets", tiles))

    # --- 2. 能见度扫描 ---
    wr.set_style(base_style)
    tiles = []
    for v in (8.0, 3.0, 1.5, 0.8):
        wr.set_params(WaterParams.from_visibility(v))
        tiles.append((f"vis={v:g}m", f"beta_R={wr.params.beta[0]:.2f}", wr.render(data, cam)))
    rows.append(("visibility sweep  —  WaterParams.from_visibility(v)", tiles))

    # --- 3. 域随机化采样 ---
    wr.set_style(base_style)
    tiles = []
    for i in range(4):
        wr.set_params(WaterParams.sample(rng, level="medium"))
        tiles.append((f"sample {i + 1}", f"vis {wr.params.visibility_m:.1f}m", wr.render(data, cam)))
    rows.append(("domain-randomized samples  —  WaterParams.sample(rng)", tiles))

    # --- 4. 光照风格 ---
    wr.set_params("coastal")
    tiles = []
    for s in ("surface", "mid", "deep", "dark"):
        wr.set_style(s)
        tiles.append((s, f"headlight x{s == 'dark'} ", wr.render(data, cam)))
    rows.append(("lighting styles  —  SceneStyle (params=coastal)", tiles))

    wr.set_style(base_style)
    wr.set_params("clear")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="underwater_vision 自检 demo")
    ap.add_argument("--xml", type=str, default=None, help="场景 XML（默认 UranUS 抓取场景）")
    ap.add_argument("--out", type=str, default=str(_HERE / "out"), help="输出目录")
    ap.add_argument("--cameras", type=str, default="grasp_cam,wrist_cam")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--style", type=str, default="surface",
                    help="基础光照风格：surface/mid/deep/dark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-depth", action="store_true",
                    help="关掉逐像素深度（用统一距离，快一倍但少了距离衰减）")
    ap.add_argument("--list-presets", action="store_true")
    args = ap.parse_args(argv)

    if args.list_presets:
        print("可用预设：")
        for n in preset_names():
            p = WaterParams.preset(n)
            print(f"  {n:9s} {p.summary()}")
        return 0

    try:
        model, data, scene_desc = load_scene(args.xml)
    except Exception as e:      # 渲染/加载失败要给出人话
        print(f"！加载场景失败：{e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    print(f"场景：{scene_desc}")
    print(f"extent={model.stat.extent:.3f}  nlight={model.nlight}  ncam={model.ncam}  "
          f"尺寸={args.size}  深度={'on' if not args.no_depth else 'off'}")

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    wr = WaterRenderer(model, args.size, args.size, params="clear", seed=args.seed,
                       use_depth=not args.no_depth, style=None)
    print(f"native fog 支持：{wr.native_fog}（mujoco 3.8 起为 False，水体光学走图像域）")
    print(f"相机 fovy：" + ", ".join(f"{c}={wr._fovy(c):g}" for c in cams))

    report: dict = {"scene": scene_desc, "camera": {}, "renderer": wr.info()}
    for cam in cams:
        try:
            rows = build_groups(wr, data, cam, args.seed, args.style)
        except Exception as e:
            print(f"  跳过相机 {cam}：{e}")
            continue

        # 单独存每张图 + 统计
        print(f"\n=== 相机 {cam} ===")
        print(f"{'标签':<20}{'mean':>7}{'R':>7}{'G':>7}{'B':>7}{'contrast':>10}{'dark%':>7}")
        base_c = None
        for row_label, tiles in rows:
            for tlabel, _sub, img in tiles:
                st = img_stats(img)
                if base_c is None:
                    base_c = max(st["contrast"], 1e-6)
                save_png(out_dir / f"{cam}_{tlabel.replace(' ', '_').replace('=', '')}.png", img)
                print(f"{tlabel:<20}{st['mean'] * 255:7.1f}{st['rgb'][0] * 255:7.1f}"
                      f"{st['rgb'][1] * 255:7.1f}{st['rgb'][2] * 255:7.1f}"
                      f"{st['contrast'] / base_c:10.2f}{st['dark_pct']:7.1f}")
        sheet = compose_sheet(rows, out_dir / f"{cam}_sheet.png",
                              f"underwater_vision demo — {Path(scene_desc).name} / {cam} "
                              f"({args.size}px, depth={'on' if not args.no_depth else 'off'})")
        print(f"拼版图：{sheet}")
        report["camera"][cam] = {"contrast_ratio": {}, "sheet": str(sheet)}

    (out_dir / "demo_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    wr.close()

    print(f"\n报告：{out_dir / 'demo_report.json'}")
    print("""
接入采集/评测（两行）：
    from underwater_vision import WaterRenderer
    wr = WaterRenderer(model, 224, 224, params="turbid", seed=0)   # 路径换成你的渲染器
    img = wr.render(data, camera="grasp_cam")                      # 输出即水下图
    # 已有 mujoco.Renderer 不想重建：WaterRenderer.wrap(my_renderer, params="turbid")
    # 每次换一段 episode 想换水况：wr.resample(level="medium")
    # dataset meta 里记：json.dump(wr.info(), f)  —— 评测端用同一个 params，别让两边成像不一致
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())