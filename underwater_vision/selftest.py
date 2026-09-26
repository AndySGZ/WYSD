# -*- coding: utf-8 -*-
"""模组自检：不靠肉眼，用数值验证成像模型 / 渲染封装 / 参数序列化都正确。

    python -m underwater_vision.selftest            # 全部检查 + 结果表
    python -m underwater_vision.selftest --ascii    # 额外打印 ASCII 亮度图（看结构）
    python -m underwater_vision.selftest --xml my_scene.xml

退出码 0 = 全过，1 = 有失败项。可以直接挂进 CI。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .demo import load_scene
from .image import apply_water, radial_factor
from .params import PRESETS, WaterParams, preset_names
from .preview import ascii_map
from .render import WaterRenderer

CAM = "grasp_cam"
_FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def contrast(img: np.ndarray) -> float:
    a = np.asarray(img, np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    return float((a @ np.array([0.299, 0.587, 0.114], np.float32)).std())


# ----------------------------------------------------------------------
def check_params() -> None:
    print("\n=== 1. 参数 / 预设 / 序列化 ===")
    check("预设齐全", set(preset_names()) == set(PRESETS), f"{preset_names()}")
    p = WaterParams.preset("harbor")
    p2 = WaterParams.from_dict(json.loads(p.to_json()))
    check("json 往返一致（hash 相同）", p2.hash() == p.hash(), p2.summary())
    check("预设摘要可读", "能见度" in p.summary() and "beta" in p.summary())

    lo = WaterParams.from_visibility(8.0)
    hi = WaterParams.from_visibility(1.0)
    check("能见度越小 beta 越大", hi.beta[0] > lo.beta[0],
          f"8m->beta_R={lo.beta[0]:.2f}  1m->beta_R={hi.beta[0]:.2f}")
    check("能见度换算自洽", abs(hi.visibility_m - 1.0) < 1e-6, f"{hi.visibility_m:.3f}m")

    s1 = WaterParams.sample(3, level="medium")
    s2 = WaterParams.sample(3, level="medium")
    check("sample 同 seed 可复现", s1.hash() == s2.hash(), s1.summary())
    check("sample 参数在合理范围",
          all(np.isfinite(v) for v in (*s1.beta, *s1.ambient)) and s1.noise_sigma < 13)

    try:
        WaterParams.preset("no-such-preset")
        ok = False
    except KeyError:
        ok = True
    check("错误预设名会报错", ok)


def check_image(clean: np.ndarray) -> None:
    print("\n=== 2. 成像模型（图像域） ===")
    zero = WaterParams(name="null", beta=(0, 0, 0), ambient=(0, 0, 0), beta_scatter=(0, 0, 0),
                       depth_max=1e6, scatter_sigma=0, noise_sigma=0, shot_noise=0,
                       particle_density=0, vignette=0, exposure=1.0, gamma=1.0,
                       jpeg_quality=0, radial_depth=False)
    ident = apply_water(clean, zero, distance=1.0, rng=0)
    check("零参数 = 恒等映射（管线无副作用）",
          np.abs(ident.astype(int) - clean.astype(int)).max() <= 1)

    a = apply_water(clean, "turbid", distance=2.0, rng=7)
    b = apply_water(clean, "turbid", distance=2.0, rng=7)
    c = apply_water(clean, "turbid", distance=2.0, rng=8)
    check("同 seed 完全一致（可复现）", np.array_equal(a, b))
    check("不同 seed 结果不同（噪声/颗粒真的在起作用）",
          not np.array_equal(a, c), f"mean|Δ|={np.abs(a.astype(int) - c.astype(int)).mean():.2f}")

    f = apply_water(clean.astype(np.float32) / 255.0, "coastal", distance=2.0, rng=0)
    check("uint8 进 uint8 出 / float 进 float 出",
          a.dtype == np.uint8 and f.dtype == np.float32 and 0.0 <= f.min() and f.max() <= 1.0)

    try:
        apply_water(np.zeros((8, 8), np.uint8), "clear")
        ok = False
    except ValueError:
        ok = True
    check("形状不对时报错清晰", ok)

    cc = [contrast(clean), contrast(apply_water(clean, "clear", distance=1.2, rng=0)),
          contrast(apply_water(clean, "coastal", distance=1.2, rng=0)),
          contrast(apply_water(clean, "turbid", distance=1.2, rng=0))]
    check("对比度随水况单调下降 clean>clear>coastal>turbid",
          all(cc[i] > cc[i + 1] for i in range(3)),
          " > ".join(f"{v:.3f}" for v in cc))

    mean0 = clean.astype(np.float32).reshape(-1, 3).mean(0)
    mean2 = apply_water(clean, "turbid", distance=1.5, rng=0).astype(np.float32).reshape(-1, 3).mean(0)
    check("红通道比蓝通道掉得快（红光先消失）",
          (mean2[0] / mean0[0]) < (mean2[2] / mean0[2]),
          f"R {mean0[0]:.0f}->{mean2[0]:.0f}   B {mean0[2]:.0f}->{mean2[2]:.0f}")
    check("水下图整体变暗但不是全黑", 5 < mean2.mean() < mean0.mean(),
          f"{mean0.mean():.1f} -> {mean2.mean():.1f}")

    rf = radial_factor(64, 64, 50.0)
    # 上边缘中点：tan = tan(25°) -> 1/cos = 1.10；角落：半对角是半高的 √2 倍 -> 1.19
    check("径向深度修正：中心=1 / 边缘中点≈1.10 / 角落≈1.19",
          abs(rf[32, 32] - 1.0) < 1e-3 and 1.09 < rf[0, 32] < 1.11 and 1.18 < rf[0, 0] < 1.20,
          f"center={rf[32, 32]:.4f} edge={rf[0, 32]:.4f} corner={rf[0, 0]:.4f}")


def check_render(model, data) -> WaterRenderer:
    print("\n=== 3. 渲染封装 WaterRenderer ===")
    import mujoco

    wr = WaterRenderer(model, 96, 96, params="clear", seed=0, use_depth=True)
    check("native fog 情况已知（3.8 起为 False）", isinstance(wr.native_fog, bool),
          f"native_fog={wr.native_fog}")

    clean = wr.clean(data, CAM)
    wr.set_params("coastal")
    # 注意：wr.rng 是有状态的，连续两次 render 会消耗不同噪声 —— 这是刻意的
    # （每帧独立噪声）。要比对就必须给显式 rng。
    g1, g2 = np.random.default_rng(0), np.random.default_rng(0)
    wet = wr.render(data, CAM, rng=g1)
    clean2, wet2 = wr.render_pair(data, CAM, rng=g2)
    check("render_pair 的清水图与 clean() 一致", np.array_equal(clean, clean2))
    check("render_pair 与 render 走同一条路径（同 rng 下逐位一致）", np.array_equal(wet, wet2))
    check("共用状态 RNG 时两帧噪声不同（每帧独立采样，符合预期）",
          not np.array_equal(wr.render(data, CAM), wr.render(data, CAM)))
    check("输出尺寸/类型正确", wet.shape == (96, 96, 3) and wet.dtype == np.uint8)
    check("render(apply=False) 等价于原始渲染",
          np.array_equal(wr.render(data, CAM, apply=False), clean))

    dep = wr.depth(data, CAM)
    far = model.vis.map.zfar * model.stat.extent
    check("深度图是米制、最大值=远裁剪面",
          dep.dtype == np.float32 and abs(float(dep.max()) - far) < 0.05,
          f"p50={np.percentile(dep, 50):.2f}m max={dep.max():.2f}m far={far:.2f}m")

    lum = clean.astype(np.float32).mean(2)
    sky = dep > far * 0.999
    check("深度/RGB 对齐：天空(远裁剪面)在清水图里是黑的",
          float(lum[sky].mean()) < float(lum[~sky].mean()) - 40,
          f"sky={lum[sky].mean():.1f} rest={lum[~sky].mean():.1f}")
    near = dep < np.percentile(dep, 6)
    rgb_near = clean.astype(np.float32)[near].mean(0)
    check("深度/RGB 对齐：最近处是黄色物块(R>B)", rgb_near[0] > rgb_near[2] + 20,
          f"RGB={rgb_near.round(1)}")

    # 逐距离衰减：只取"亮像素"，看存活率是否随距离单调下降
    p = WaterParams.preset("coastal", radial_depth=False)
    w = apply_water(clean, p, depth=dep, rng=0).astype(np.float32)
    c = clean.astype(np.float32)
    bright = lum > 100
    edges = [0.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]
    ratios, rows = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = bright & (dep >= lo) & (dep < hi)
        if m.sum() < 20:
            continue
        r = float(w[m].mean() / max(c[m].mean(), 1e-6))
        ratios.append(r)
        rows.append((f"[{lo:g},{hi:g})", int(m.sum()), float(c[m].mean()), r))
    print(f"  {'深度桶(m)':<14}{'亮像素数':>9}{'clean均值':>11}{'存活率':>9}")
    for label, n, cm, r in rows:
        print(f"  {label:<14}{n:>9}{cm:>11.1f}{r:>9.3f}")
    check("亮像素存活率随距离单调下降（衰减随深度生效）",
          all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1)),
          " > ".join(f"{v:.2f}" for v in ratios))

    # 全黑背景 + 背向散射 = 水色幕（水下最直观的现象）
    w_sky = w[sky].mean(0)
    check("全黑背景被背向散射抬成水色（雾幕）",
          float(w_sky.mean()) > float(c[sky].mean()) + 10,
          f"clean={c[sky].mean():.1f} -> wet={w_sky.mean():.1f} RGB={w_sky.round(1)}")
    check("背景水色接近 ambient（偏蓝绿）",
          w_sky[2] > w_sky[0] * 1.5, f"ambient={p.ambient_rgb * 255}")

    # 多相机 / wrap
    many = wr.render_many(data, {"top": CAM, "wrist": "wrist_cam"})
    check("render_many 多相机", set(many) == {"top", "wrist"}
          and all(v.shape == (96, 96, 3) for v in many.values()))
    r = mujoco.Renderer(model, 96, 96)
    wr2 = WaterRenderer.wrap(r, params="turbid", seed=0)
    check("WaterRenderer.wrap() 复用已有 Renderer（不新开 GL 上下文）",
          wr2.render(data, CAM).shape == (96, 96, 3) and wr2.model is model)
    check("fovy 自动读取（grasp_cam=50）", abs(wr2._fovy(CAM) - 50.0) < 1e-6)
    r.close()

    # 光照风格幂等
    wr3 = WaterRenderer(model, 96, 96, params="coastal", seed=0, style="deep", use_depth=False)
    m1 = float(wr3.render(data, CAM).mean())
    wr3.set_style("deep")
    wr3.set_style("surface")
    wr3.set_style("deep")
    m2 = float(wr3.render(data, CAM).mean())
    check("反复 set_style 不累积（灯光走快照绝对赋值）", abs(m1 - m2) < 2.0,
          f"mean {m1:.2f} vs {m2:.2f}")

    info = wr3.info()
    check("info() 可直接写进 dataset meta",
          {"water_params", "water_hash", "seed", "use_depth"} <= set(info),
          f"hash={info['water_hash']}")

    check("域随机化：resample 换水况且同 seed 可复现",
          WaterParams.sample(0, level="medium").hash()
          == WaterParams.sample(0, level="medium").hash())
    wr3.close()
    return wr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="underwater_vision 自检")
    ap.add_argument("--xml", type=str, default=None)
    ap.add_argument("--ascii", action="store_true", help="打印 ASCII 亮度图")
    args = ap.parse_args(argv)

    try:
        model, data, desc = load_scene(args.xml)
    except Exception as e:
        print(f"！场景加载失败：{e}", file=sys.stderr)
        return 2
    print(f"场景：{desc}\nextent={model.stat.extent:.3f} nlight={model.nlight} ncam={model.ncam}")

    check_params()
    wr = WaterRenderer(model, 96, 96, params="clear", seed=0)
    clean = wr.clean(data, CAM)
    check_image(clean)

    if args.ascii:
        print("\n--- 清水图 ---\n" + ascii_map(clean))
        for name in ("coastal", "turbid"):
            print(f"\n--- {name} ---\n" + ascii_map(apply_water(clean, name, distance=1.6, rng=0)))
    wr.close()

    wr2 = check_render(model, data)
    wr2.close()

    print("\n" + "=" * 46)
    if _FAILED:
        print(f"FAILED {len(_FAILED)} 项：{_FAILED}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())