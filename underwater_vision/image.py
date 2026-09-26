# -*- coding: utf-8 -*-
"""水下成像模型（图像域）：把 MuJoCo 渲染出来的"清水图"变成"水下图"。

只用 numpy（颗粒/噪声/模糊全部自己实现，不依赖 scipy/PIL；只有 JPEG 伪影那一步
才可选地用 PIL）。**不依赖 mujoco**，所以可以单独测试。

模型（Jaffe–McGlamey 简化式，逐像素、逐通道）：

    I_c = J_c · exp(-beta_c · d)  +  ambient_c · (1 - exp(-beta_s_c · d))
          └── 直接透射：衰减 + 偏色        └── 背向散射：悬浮颗粒被照亮

再依次叠加：前向散射（随距离增大的模糊）→ 曝光/白平衡/暗角 → 悬浮颗粒 → 传感器噪声。

为什么放在图像域而不是渲染器里（重要）
--------------------------------------
1. 本机 mujoco 3.8 的 classic renderer **已经不支持 fog/haze**（``MjvScene`` 里没有 fog
   字段，``mjRND_FOG``/``mjRND_HAZE`` 打开也无效，已实测），渲染器只给"清水图"。
2. 即使支持，渲染器的雾也是"统一灰雾"：不知道波长（做不到红光先丢）、不知道悬浮颗粒
   （做不到背向散射）、更没有传感器噪声。而 VLA 吃的是 2D 图像 —— 在图像域做水的域
   随机化，自由度最大、成本最低、和训练/评测一致性最好。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .params import WaterParams

__all__ = [
    "apply_water",
    "make_water_fn",
    "distance_map",
    "radial_factor",
    "gaussian_blur",
]

# ----------------------------------------------------------------------
# 兼容前端：WaterParams / 预设名 / dict / None
# ----------------------------------------------------------------------
def _resolve_params(params: Any) -> WaterParams:
    if params is None:
        return WaterParams.preset("clear")
    if isinstance(params, WaterParams):
        return params
    if isinstance(params, str):
        return WaterParams.preset(params)
    if isinstance(params, dict):
        return WaterParams.from_dict(params)
    raise TypeError(f"params 只能是 WaterParams / 预设名(str) / dict / None，收到 {type(params)}")


def _as_float(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """返回 (0~1 float32 图, 输入是否是 uint8)。"""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0, True
    return a.astype(np.float32), False


# ----------------------------------------------------------------------
# 距离图
# ----------------------------------------------------------------------
def radial_factor(height: int, width: int, fovy_deg: float) -> np.ndarray:
    """把"沿光轴的 z 深度"修正成"径向距离"的乘子 1/cos(theta)。

    MuJoCo 的深度是 OpenGL z（沿光轴），而水的衰减是沿光线（径向）的。
    对 fovy=50° 的相机，画面边缘的修正量约 10% —— 不大，但白送。
    """
    tan_half = np.tan(np.deg2rad(fovy_deg) / 2.0)
    yy = (np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / (height / 2.0)
    xx = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / (height / 2.0)
    tan2 = (yy[:, None] ** 2 + xx[None, :] ** 2) * (tan_half**2)
    return np.sqrt(1.0 + tan2)


def distance_map(shape: tuple[int, int], depth: np.ndarray | None = None,
                 params: WaterParams | None = None, distance: float | None = None,
                 fovy_deg: float | None = None, radial: bool | None = None) -> np.ndarray:
    """产出 (H,W) float32 的"水中的传播距离"。

    优先级：``depth``（逐像素） > ``distance``（统一值） > ``params.depth_default``。
    形状不一致的深度图会做最近邻缩放（不引 scipy）。
    """
    p = params or WaterParams.preset("clear")
    h, w = int(shape[0]), int(shape[1])
    do_radial = p.radial_depth if radial is None else bool(radial)

    if depth is not None:
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        if d.shape[:2] != (h, w):
            yi = (np.arange(h, dtype=np.float32) * (d.shape[0] / h)).astype(np.int64)
            xi = (np.arange(w, dtype=np.float32) * (d.shape[1] / w)).astype(np.int64)
            d = d[yi][:, xi]
        d = d * float(p.depth_scale)
        # 背景（远裁剪面）可能是 zfar 值，也可能是 0（取决于 readDepthMap 约定）：
        # 非有限值与非正值统一按"最远"处理。
        d = np.where(np.isfinite(d) & (d > 0), d, p.depth_max).astype(np.float32)
        if do_radial and fovy_deg:
            d = d * radial_factor(h, w, float(fovy_deg))
    elif distance is not None:
        d = np.full((h, w), float(distance), dtype=np.float32)
    else:
        d = np.full((h, w), float(p.depth_default), dtype=np.float32)

    return np.clip(d, float(p.depth_min), float(p.depth_max)).astype(np.float32)


# ----------------------------------------------------------------------
# 纯 numpy 模糊（三次 box blur 近似高斯）
# ----------------------------------------------------------------------
def _slice_axis(a: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(start, stop)
    return a[tuple(idx)]


def _box_blur_axis(x: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius < 1:
        return x
    n = x.shape[axis]
    k = 2 * radius + 1
    pad = [(0, 0)] * x.ndim
    pad[axis] = (radius, radius)
    xp = np.pad(x, pad, mode="reflect")
    c = np.cumsum(xp, axis=axis, dtype=np.float32)
    zeros_shape = list(xp.shape)
    zeros_shape[axis] = 1
    c0 = np.concatenate([np.zeros(zeros_shape, dtype=np.float32), c], axis=axis)
    return (_slice_axis(c0, axis, k, k + n) - _slice_axis(c0, axis, 0, n)) / k


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """三次 box blur 近似的高斯模糊（纯 numpy，O(N)）。"""
    if sigma <= 0:
        return img
    # 3 次宽度 w 的 box 滤波 ≈ 高斯 σ，σ² = 3(w²-1)/12
    w = int(max(3, round(np.sqrt(12.0 * sigma * sigma / 3.0 + 1.0))))
    r = max(1, (w - 1) // 2)
    out = np.asarray(img, dtype=np.float32)
    for _ in range(3):
        for axis in (0, 1):
            out = _box_blur_axis(out, r, axis)
    return out


# ----------------------------------------------------------------------
# 悬浮颗粒 / 暗角 / 噪声 / JPEG
# ----------------------------------------------------------------------
def _add_particles(img: np.ndarray, density: float, brightness: float,
                   rng: np.random.Generator, ambient: np.ndarray) -> np.ndarray:
    """海雪 / 泥沙悬浮颗粒：随机亮斑。density = 每千像素颗粒数。"""
    h, w = img.shape[:2]
    n = int(round(float(density) * h * w / 1000.0))
    if n <= 0 or brightness <= 0:
        return img
    out = img
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    sig = rng.uniform(0.7, 2.2, n).astype(np.float32)
    amp = (rng.uniform(0.3, 1.0, n) * float(brightness)).astype(np.float32)
    color = np.clip(0.75 + ambient, 0.0, 1.0).astype(np.float32)   # 偏白、带一点水色
    for y, x, s, a in zip(ys, xs, sig, amp):
        r = int(np.ceil(2.5 * s))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy = (np.arange(y0, y1, dtype=np.float32) - y)
        xx = (np.arange(x0, x1, dtype=np.float32) - x)
        g = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * s * s))
        out[y0:y1, x0:x1] += (a * g)[..., None] * color
    return out


def _vignette_map(h: int, w: int, strength: float) -> np.ndarray:
    yy = (np.arange(h, dtype=np.float32) + 0.5 - h / 2.0) / (h / 2.0)
    xx = (np.arange(w, dtype=np.float32) + 0.5 - w / 2.0) / (w / 2.0)
    r2 = yy[:, None] ** 2 + xx[None, :] ** 2
    return 1.0 - float(strength) * np.clip(r2 / 2.0, 0.0, 1.0)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    try:
        import io

        from PIL import Image
    except ImportError:
        return img   # 没装 PIL 就跳过（不报错，流水线不能因此挂掉）
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def apply_water(rgb: np.ndarray, params: Any = None, depth: np.ndarray | None = None,
                distance: float | None = None, rng: np.random.Generator | int | None = None,
                fovy_deg: float | None = None, radial: bool | None = None,
                out_dtype: str | np.dtype | None = None) -> np.ndarray:
    """把清水渲染图变成水下图。

    Args:
        rgb: (H,W,3) uint8 或 0~1 float。返回类型默认与输入一致。
        params: :class:`WaterParams` / 预设名（``"turbid"``…）/ dict / None(=clear)。
        depth: (H,W) 米制深度（MuJoCo ``Renderer`` 开深度渲染即得）。给了就逐像素算衰减。
        distance: 不给深度图时的统一距离（米）。
        rng: 随机源（颗粒/噪声/白平衡）。要可复现就传 ``np.random.default_rng(seed)``。
        fovy_deg: 相机垂直视场角（度），用于把 z 深度修正成径向距离。
        radial: 覆盖 ``params.radial_depth``。
        out_dtype: ``"uint8"`` / ``"float32"``，默认跟随输入。

    Returns:
        同形状的水下图。
    """
    p = _resolve_params(params)
    img, was_uint8 = _as_float(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"rgb 需要 (H,W,3)，收到 {img.shape}")
    h, w = img.shape[:2]

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    d = distance_map((h, w), depth=depth, params=p, distance=distance,
                     fovy_deg=fovy_deg, radial=radial)

    # --- 1. 直接透射 + 背向散射 ---
    beta = p.beta_rgb[None, None, :]
    beta_s = np.asarray(p.scatter_beta, dtype=np.float32)[None, None, :]
    d3 = d[..., None]
    trans = np.exp(-beta * d3)
    trans_s = np.exp(-beta_s * d3)
    out = img * trans + p.ambient_rgb[None, None, :] * (1.0 - trans_s)

    # --- 2. 前向散射：越远越糊 ---
    if p.scatter_sigma > 0:
        blurred = gaussian_blur(out, float(p.scatter_sigma))
        wt = np.clip(d / max(float(p.scatter_dist), 1e-6), 0.0, 1.0)[..., None]
        out = out * (1.0 - wt) + blurred * wt

    # --- 3. 相机侧：曝光 / gamma / 白平衡 / 暗角 ---
    if p.exposure != 1.0:
        out = out * float(p.exposure)
    if p.gamma != 1.0:
        out = np.clip(out, 0.0, 1.0) ** float(p.gamma)
    if not np.allclose(p.wb_rgb, 1.0):
        out = out * p.wb_rgb[None, None, :]
    if p.vignette > 0:
        out = out * _vignette_map(h, w, p.vignette)[..., None]

    # --- 4. 悬浮颗粒 ---
    if p.particle_density > 0 and p.particle_brightness > 0:
        out = _add_particles(out, p.particle_density, p.particle_brightness, rng, p.ambient_rgb)

    # --- 5. 传感器噪声 ---
    if p.noise_sigma > 0:
        out = out + rng.normal(0.0, float(p.noise_sigma) / 255.0,
                               size=out.shape).astype(np.float32)
    if p.shot_noise > 0:
        out = out + (rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
                     * np.sqrt(np.clip(out, 0.0, None)) * float(p.shot_noise))
    if p.jpeg_quality > 0:
        out = _jpeg_roundtrip(out, p.jpeg_quality)

    out = np.clip(out, 0.0, 1.0)

    if out_dtype is not None:
        return (out * 255.0).astype(np.uint8) if str(out_dtype) == "uint8" else out.astype(np.float32)
    return (out * 255.0).astype(np.uint8) if was_uint8 else out.astype(np.float32)


def make_water_fn(params: Any = None, **kw) -> Callable[..., np.ndarray]:
    """把一组固定参数包成一个函数，方便挂进采集/评测回路：

        wet = make_water_fn("turbid", fovy_deg=50)
        img = wet(clean_rgb, depth=depth_map)
    """
    p = _resolve_params(params)

    def _fn(rgb: np.ndarray, depth: np.ndarray | None = None,
            rng: np.random.Generator | int | None = None, **over) -> np.ndarray:
        return apply_water(rgb, params=p, depth=depth, rng=rng, **{**kw, **over})

    _fn.params = p        # type: ignore[attr-defined]
    return _fn