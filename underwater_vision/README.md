# underwater_vision — MuJoCo 水下视觉渲染模组

把 MuJoCo 渲染出来的**清水图**过一遍水下成像模型，得到**水下图**；并附带水下光照（探照灯/环境光）
设置与域随机化采样。为 **UranUS 水下机械臂 VLA 微调**服务（见
`WYSD/Lerobot-Uranus-VLA-Demo/docs/underwater_vla_task_design.md` 的 S1 阶段）。

## 放在哪 / 怎么 import

本模组在 `AndySGZ/WYSD` 仓库里位于**仓库根目录**：`WYSD/underwater_vision/` —— **唯一来源**
（开发时曾在仓库外 `E:/projects/mujoco/underwater_vision`，2026-09 已合并回来并删除旧副本）。

```powershell
$env:PYTHONPATH = "E:\projects\mujoco\WYSD"       # 仓库根目录即可 import
```

`paths.py` 会从本文件所在目录**向上逐层探测** `Lerobot-Uranus-VLA-Demo`，所以万一以后又被
放到别的位置也能自动找到 UranUS 场景；也可以用环境变量 `UNDERWATER_VISION_DEMO_DIR`
直接指定 demo 仓库路径。

## 最短用法（两行）

```python
from underwater_vision import WaterRenderer

wr = WaterRenderer(model, 224, 224, params="turbid", seed=0)   # 一行
img = wr.render(data, camera="grasp_cam")                      # 输出即水下图
```

要"清水 / 水下"成对拿（做消融）：

```python
clean, wet = wr.render_pair(data, "grasp_cam")
```

不需要 mujoco，只处理已有图像：

```python
from underwater_vision import apply_water, WaterParams
wet = apply_water(clean_rgb, WaterParams.preset("coastal"), depth=depth_map, rng=0)
```

## 为什么水体光学放在图像域（本机 mujoco 3.8.0 实测）

| 结论 | 证据 |
|---|---|
| classic renderer 的 **fog/haze 已失效** | `MjvScene` 里**没有** fog 字段；`vis.map.fogstart/fogend/haze` 与 `rgba.fog/haze` 仍可赋值但改了没有任何像素变化；`mjRND_FOG`/`mjRND_HAZE` 打开同样无效 |
| **灯光仍然生效** | `vis.headlight.{ambient,diffuse,specular,active}` 与 `model.light_*`（pos/dir/diffuse/cutoff/attenuation/castshadow）实测有效 |
| **深度渲染可用** | `Renderer.enable_depth_rendering()` 返回米制深度 (H,W) float32，背景 ≈ 远裁剪面（`vis.map.zfar × stat.extent`），与 RGB 逐像素对齐 |

即使渲染器的雾能用，它也表达不了"红光先丢 / 背向散射 / 悬浮颗粒 / 传感器噪声"。
而 VLA 吃的是 2D 图像 —— 图像域做水的域随机化，自由度最大、最便宜、训练与评测最容易保持一致。

## 目录

```
underwater_vision/
├── params.py     # WaterParams：参数 / 预设 / 能见度换算 / 域随机化采样（只依赖 numpy）
├── image.py      # apply_water：水下成像模型（只依赖 numpy）
├── scene.py      # SceneStyle：探照灯/环境光/场景灯 + XML 片段（不 import mujoco）
├── render.py     # WaterRenderer：包装 mujoco.Renderer（最常用入口）
├── demo.py       # 自检 demo：渲染对照图（python -m underwater_vision.demo）
├── animate.py    # 报告用动画：多视角 + 中文注解，出 GIF/MP4（python -m underwater_vision.animate）
├── selftest.py   # 数值自检（python -m underwater_vision.selftest）
├── preview.py    # 终端看图：把图像打成 ASCII 亮度图（无 GUI 时用）
├── out/          # demo 输出
└── out/anim/     # 动画输出（GIF / MP4 / 静态图）
```

依赖：**numpy 必需**；渲染部分需要 **mujoco**；`demo.py` / `save_png` / JPEG 伪影需要 **PIL**。
本仓库的 `mujoco-env` venv 里三者都有；`指南.md` 已把 `E:\projects\mujoco\WYSD`（含
`E:\projects\mujoco`）加进 `PYTHONPATH`，所以任何脚本都能直接 `import underwater_vision`。

## 成像模型

逐像素、逐通道（Jaffe–McGlamey 简化式）：

```
I_c = J_c · exp(-β_c · d)  +  ambient_c · (1 - exp(-β_s,c · d))
      └── 直接透射：衰减 + 偏色        └── 背向散射：悬浮颗粒被照亮
```

再依次叠加：**前向散射**（越远越糊）→ **曝光/gamma/白平衡/暗角** → **悬浮颗粒（海雪）** → **传感器噪声（高斯+散粒）→ JPEG 伪影**。

`d` 的来源（`distance_map()` 的优先级）：
`depth`（逐像素，来自 `Renderer` 深度渲染）> `distance`（统一值）> `params.depth_default`。
`depth` 会被截断到 `[depth_min, depth_max]`，`radial_depth=True` 时还会把"沿光轴的 z 深度"
按 `1/cosθ` 修正成"径向距离"（fovy=50° 时边缘中点 ×1.10、角落 ×1.19，已实测）。

## 参数速查（`WaterParams`）

| 字段 | 含义 | 备注 |
|---|---|---|
| `beta` | 分通道衰减 (R,G,B)，1/m | 红 > 绿 > 蓝；"红光先消失"就靠它 |
| `beta_scatter` | 背向散射系数；`None` = 跟随 `beta` | 想单独调"雾幕厚度"就改它 |
| `ambient` | 无限远水色 / 背向散射色 | 背景被抬成的颜色 |
| `depth_default` / `depth_min` / `depth_max` | 无深度图时的距离 / 深度截断范围 | `depth_max` 决定"最远能看见多少" |
| `depth_scale` | 深度整体缩放 | 相机/尺度标定用 |
| `radial_depth` | z 深度 → 径向距离修正 | 默认 True |
| `scatter_sigma` / `scatter_dist` | 前向散射最大模糊 σ(px) / 达到最大模糊的距离 | 0 = 关 |
| `exposure` / `gamma` / `wb_gain` | 曝光 / 幂律 / 白平衡漂移 | 模拟自动白平衡在蓝绿环境失准 |
| `vignette` | 暗角强度 0~1 | 探照灯边缘暗区 |
| `noise_sigma` / `shot_noise` | 高斯读出噪声（/255）/ 散粒噪声 | 低照度噪声 |
| `jpeg_quality` | 0=不做；否则 JPEG 往返 | 需要 PIL |
| `particle_density` / `particle_brightness` | 悬浮颗粒密度（每千像素）/ 亮度 | 海雪、泥沙 |

### 预设

`preset_names()` → `clear` / `coastal` / `turbid` / `deep` / `harbor`。
注意 **`deep` 是"深海颜色"（很蓝）而不是"很浊"**，它的衰减反而比 `turbid` 小。
"能见度"这个更直观的旋钮用 `WaterParams.from_visibility(v_m)`（距离 v 米处对比度降到 ~5%）。

三种拿参数的方式：

```python
WaterParams.preset("turbid")              # 预设
WaterParams.from_visibility(1.5)          # 用"能见度 1.5m"描述
WaterParams.sample(rng, level="medium")   # 域随机化（easy/medium/hard 缩放难度）
```

`p.to_json(path)` / `WaterParams.from_json(path)` / `p.hash()` 用于**保证采集端与评测端成像一致**。

## 接入现有 UranUS 采集/评测（各改一行）

### `collect_data_vla.py`

```python
from underwater_vision import WaterRenderer                      # ① 加 import
...
renderer = WaterRenderer(env.model, size, size, params=args.water,  # ② 换构造
                         seed=args.seed, use_depth=not args.no_depth)
...
def grab_images():
    imgs = {}
    for key, cam in (("observation.images.top", args.top_cam),
                     ("observation.images.wrist", args.wrist_cam)):
        imgs[key] = renderer.render(env.data, camera=cam)         # ③ 原来两行变一行
    frame_imgs.append(imgs)
```

再加一句把水参数写进 dataset 目录，评测端照着读：

```python
(Path(args.out_root).parent / f"{Path(args.out_root).name}_water.json").write_text(
    renderer.params.to_json(), encoding="utf-8")
```

### `eval_vla.py`

```python
from underwater_vision import WaterRenderer, WaterParams

class ObsBuilder:
    def __init__(self, env, size, cams, device, water_json=None):
        ...
        self.water = WaterRenderer.wrap(self.renderer,      # 复用已有 Renderer，不新开 GL 上下文
                                       params=WaterParams.from_json(water_json) if water_json
                                       else "turbid", seed=0)

    def images(self):
        out = {}
        for key, cam in self.cams:
            img = self.water.render(self.env.data, camera=cam)   # 换这一行
            out[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        return out
```

> ⚠️ **训练与评测必须用同一组水参数**。把 `wr.info()` 写进 dataset meta，
> 评测时读回来；`WaterParams.hash()` 不同就说明两边成像不一致，实验作废。

### 域随机化（可选）

```python
wr = WaterRenderer(model, 224, 224, params="coastal", seed=0, style="mid")
for ep in range(n_episodes):
    wr.resample(level="medium")        # 每段换一组水况
    wr.randomize_scene_lights(dir_jitter_deg=10)   # 灯光方向/强度抖动
    ...
```

注意 `LeRobotDataset.create()` **不能续采**，而水参数若每段都变就必须逐段记录 ——
所以 S1 消融建议**每个 dataset 固定一档水况**（清水一套、浊水一套），
只在同一档内做小幅随机化。

## 自检

```bash
python -m underwater_vision.selftest          # 全部数值检查，退出码 0/1，可挂 CI
python -m underwater_vision.selftest --ascii  # 额外打印 ASCII 亮度图（看结构）
python -m underwater_vision.demo              # 渲染对照图到 ./out/（含拼版图 + 报告）
python -m underwater_vision.demo --list-presets
python -m underwater_vision.preview out/*.png # 无 GUI 时在终端"看图"（ASCII + 统计）
```

`preview` 是排查利器 —— 当你看不到图（远程/无 GUI/模型不支持图像输入）时，
它能直接把渲染结果的**结构**打出来。以 UranUS 抓取场景为例（72×22 字符）：

```
clean (no water)   mean=144.1  contrast=0.280     coastal   mean=98.9  contrast=0.156     turbid   mean=67.7  contrast=0.080
++*+++++++...====  结构清晰：台面/机械臂/方块都在   -+=--:::...  结构还在但发闷、明显偏青   .......:.:.  结构几乎糊平，只剩一点轮廓
```

对比度 0.280 → 0.156 → 0.080，正是"水下视觉退化"的量化形式。

`selftest` 覆盖：零参数恒等性、同 seed 可复现、uint8/float dtype 保持、
对比度随水况单调下降、红通道比蓝通道掉得快、**亮像素存活率随距离单调下降**、
全黑背景被背向散射抬成水色、深度/RGB 逐像素对齐（天空最暗、最近处是黄色物块）、
`wrap()` 复用渲染器、反复 `set_style` 不累积、参数 json 往返、`info()` 可写 meta 等。

## 报告用动画

```bash
python -m underwater_vision.animate --all            # 三张（推荐）
python -m underwater_vision.animate --list           # 看有哪些
python -m underwater_vision.animate --anim water     # 只出"不同水况"那张
```

一张动画 = **多路相机画面并排 + 中文注解**（标题/每格说明/逐帧状态/页脚参数 hash），
每张同时出 `.gif`、`.mp4` 和一张 `_still.png`（注解与动画一致，可直接当报告插图）。

| 名字 | 画面 | 说明 |
|---|---|---|
| `task` | 3 路：grasp_cam / wrist_cam / 第三人称自由相机 | 抓取-放置专家轨迹，逐帧显示阶段与状态量 |
| `water` | 4 路：clean / coastal / turbid / harbor | **同一段运动**，只有水况在变（消融图） |
| `light` | 4 路：surface / mid / deep / dark | 水况固定，只有光照在变 |

运动源是**真实的`pick_place`专家轨迹**：采一次（3640 物理步 / 145 拍 / 抬升 140mm / 成功），
把每拍的 `qpos` 存下来，之后只做**正运动学 + 渲染**，所以三张动画的运动逐位一致 ——
变量只有水况/机位/光照，对比才站得住。

实测数字（`--size 200 --max-frames 72 --stride 2`，72 帧 / 7.2s @10fps）：

| | 分辨率 | GIF | MP4 |
|---|---|---|---|
| `task` | 632×400 | 6.7 MB | 1.2 MB |
| `water` | 840×400 | 8.6 MB | 2.5 MB |
| `light` | 840×400 | 8.1 MB | 1.4 MB |

**报告里优先嵌 MP4**：色彩无损、体积只有 GIF 的 1/4~1/6。GIF 偏大是因为逐像素
颗粒噪声（`noise_sigma`/marine snow）几乎不可压缩 —— 这不是 bug，是 LZW 的天性；
要小就 `--gif-colors 64 --stride 4`。

常用旋钮：`--size`（单格边长，默认 200）、`--max-frames`（默认 72）、`--stride`
（每几拍取一帧，2 → 10fps 实时）、`--formats gif,mp4`、`--motion expert|procedural`
（UranUS 环境不可用时自动退回程序化摆动并告警）。

> 一个坑：`task`/`water` 两张图里光照**不是**自变量，用的是 `surface`。
> 曾经全局套 `deep`（环境光 0.04），结果腕部相机对比度掉到 0.030、整格几乎全黑 ——
> 三视角报告图要的是"看得清"，暗端交给 `light` 那张图去展示。

## 已知限制与注意

1. **RNG 是有状态的**：连续两次 `render()` 会消耗不同噪声（每帧独立采样，刻意如此）。
   要逐位复现就传显式 `rng=np.random.default_rng(seed)`。整个数据集的 `seed` 记在 `info()` 里。
2. **`use_depth=True` 会让每帧多渲一遍深度图**（约慢一倍）。想快就 `use_depth=False`，
   用统一距离（`depth_default`）——代价是没有"近处清楚、远处糊"的距离层次。
3. **灯光缩放基于 `snapshot_lights()` 快照**：`WaterRenderer` 内部已处理，所以反复
   `set_style()` 不会累积。但如果你**直接**调 `scene.apply_style()` / `scene.randomize_lights()`，
   请把 `base=` 传进去，否则强度会在当前值上反复相乘。
4. **`set_viewer_fog()` 在 mujoco ≥3.8 不产生画面变化**（保留给老版本/viewer 观感）。
5. **`particle_density` 是 Python 循环逐颗画**：224² 下 density=2 约 100 颗，开销可忽略；
   但如果要用 1024² 大图 + 极高颗粒密度，注意这一步会变慢。
6. 只做了**成像**，没做水体动力学（浮力/阻力），也不做折射/焦散 —— 见设计文档 §8.1：
   不改动力学的前提下可以用"关节阻尼 / 末端负载 / 控制频率降低"当**难度旋钮**近似。

## 输出示例（UranUS 抓取场景，224px，逐像素深度）

对比度 = 亮度标准差相对清水图的比值：

| 水况 | 相对对比度 | 均值 R / G / B |
|---|---|---|
| clean | 1.00 | 140 / 146 / 145 |
| clear | 0.81 | 95 / 128 / 131 |
| coastal | 0.56 | 60 / 114 / 123 |
| turbid | 0.28 | 28 / 83 / 94 |
| harbor | 0.22 | 25 / 58 / 64 |
| vis=0.8m | 0.16 | 14 / 59 / 75 |
| dark（关灯） | 0.09 | 11 / 23 / 23 |

红通道塌得比蓝通道快得多 —— 这正是水下视觉最典型的退化，也是"黄色方块变灰"的原因。