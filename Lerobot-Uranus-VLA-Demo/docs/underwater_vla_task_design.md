# 水下机械臂 VLA 微调：任务与场景设计文档

> 定位：在 `Lerobot-Uranus-VLA-Demo`（UranUS 臂 + LeRobotDataset + π0/SmolVLA 微调链路）
> 现有成果之上，规划设计"水下"这条线要做什么任务、要造什么环境、怎么验证。
>
> **前提约定**：不考虑水下水动力学（不做流体、浮力、附加质量、水阻的物理仿真）。
> 本文只设计 **观测分布**（水的视觉）与 **语言–任务分布**（干什么活）这两件事。
>
> 相关文档：`README.md`（仓库现状）、`VLA_FINETUNE.md`（硬件门槛与微调命令）。
> 本文是**设计**，不含实现；实现时以本文的"工程落地清单"（第 7 节）逐项推进。

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 |
|---|---|
| 水下这件事主要改什么？ | 只改两个旋钮：**观测分布**（水的视觉退化）与**语言–任务分布**（作业任务族）。动力学不动。 |
| 环境怎么造？ | 分 4 层加：L0 场景几何 → L1 光照/水体渲染 → L2 域随机化 → L3 与任务耦合的能见度。**不要一次全上**。 |
| 渲染在哪做？ | **MuJoCo 原生只做灯光**：实测 mujoco 3.8 的 classic renderer 已**不支持 fog/haze**（`MjvScene` 无 fog 字段、`mjRND_FOG/HAZE` 无效），只有 `headlight` 与 `<light>` 生效。所以**分通道衰减、背向散射、散射模糊、噪声全部在图像域**做（Jaffe–McGlamey 模型）。已经在 `WYSD/underwater_vision/`（本仓库根目录）落地（见 §2.3）。不要在 MuJoCo 里做折射/焦散。 |
| 先做哪个任务？ | 按性价比：**S1 水下成像消融**（改渲染、任务不动，立刻出一张对比图）→ **S3 阀门旋转 / 探针点测**（判据二值、专家好写、接触丰富）→ **S4 hot stab 插入**（真难点）→ **S5 多任务 + 语言消融**。 |
| 什么最能证明"训好了 VLA"？ | **S5 的指令替换实验**：同场景同状态，只换指令，看动作是否改变。指令换了动作不变 = 语言分支根本没被用上。 |
| 最大的技术风险？ | ①水下视觉退化导致专家本身失败 → 采到的数据全是失败样本；②动作-观测耦合（扬尘）导致任务变成 POMDP，单帧 BC 学不会；③多任务摊薄数据量。 |
| 绝对不要做？ | ①在 MuJoCo 里做折射/体积光/焦散（相机标定与几何全崩）；②只加一层青绿色滤镜就说"这是水下"（实验价值≈0）；③一次堆 20 个任务。 |

---

## 1. 设计框架：把"任务"拆成 4 个可独立调的旋钮

VLA 微调里"训练任务"不是一个标量，而是四个可分别设计的分布：

| 旋钮 | 现有取值（已落地 ✅） | 水下要动的地方 |
|---|---|---|
| **观测分布** | `observation.images.top`(224² 彩色) + `observation.images.wrist` + `observation.state`(7) + `task`(str)。**状态里刻意不含物块位姿**，逼策略看图（`--privileged_state` 才加，作对照） | 水的成像效应、探照灯光照、低对比度场景、相机被自身/工具遮挡 |
| **语言–任务分布** | 硬编码单句 `put the yellow cube on the blue disc`（`collect_data_vla.py::DEFAULT_TASK`，`eval_vla.py::TASK`） | 指令模板池、多任务、指称方式（颜色/位置/功能） |
| **动作空间 / 本体** | `action`(7) = 目标关节 1..6 + 夹爪命令；8 执行器 = 8 关节；20Hz；`action` = 未来 `lookahead=4` 拍的实测状态 | 基本不改。可选加"低频 / 延迟 / 负载"作为**任务难度旋钮**（见 8.1） |
| **评测判据** | `lift > MIN_LIFT(2cm) and env.success()` | 每个新任务都要有"几行代码能判"的判据（见 3.4） |

**设计纪律**：任何一个新任务，必须同时定义这 4 个旋钮，否则做出来没法训练也没法评测。第 3.2 节给了填空模板。

---

## 2. 水下环境设计

### 2.1 分层策略（L0 → L3，逐层加）

| 层 | 内容 | 实现位置 | 训练上考察什么 |
|---|---|---|---|
| **L0 场景几何** | 海床/沉积物底面、弱纹理岩石、防沉板/结构件、管道、阀盘、生物附着物、缆绳、工具篮、悬浮颗粒物 | 新建 `asset/scene_uranus_underwater_*.xml` | 目标不再"孤零零放在干净桌面上"，有背景干扰、遮挡与杂乱 |
| **L1 光照 / 水体渲染** | 雾、分通道颜色衰减、背向散射、探照灯热点、低对比度、自遮挡 | XML `<visual>` + 新增 `src/water_image.py` | **视觉退化下的鲁棒性 —— 这才是水下的真难点** |
| **L2 域随机化** | L0/L1 的参数变成分布，每次 reset 采样一组 | `src/water.py` 的采样器 + dataset meta 记录 | sim→real、参数敏感性曲线 |
| **L3 任务耦合随机化** | 能见度直接决定任务可行性（低能见度时专家也会失败） | 环境 reset 时联动 | **必须和任务一起调**，否则采到的全是失败数据 |

**顺序建议**：先 L1（改渲染即可，不动任务，收益最大）→ 再 L0（换场景）→ 再 L2 → 最后 L3。

### 2.2 MuJoCo 原生能做什么（字段已在本仓库 mujoco 的 `introspect/structs.py` 中核对）

现有 `asset/scene_uranus_grasp.xml` 的 `<visual>` 块只有三行（`offwidth/offheight`、亮 `headlight`、`shadowsize`、`map znear`），扩展点很明确：

```xml
<visual>
  <!-- 雾。fogstart / fogend / znear / shadowclip 都是 × mjModel.stat.extent 的归一化值，
       Uranus 工作半径 1.2~2.0m，scene 的 extent 比教程桌面场景大得多：
       **必须重标定，不要照抄教程数值**，否则雾会跑到画面外。 -->
  <map fogstart="0.5" fogend="6" haze="0.5" znear="0.02" zfar="30"/>
  <!-- 雾色 = 水的"无限远背景色"（青绿/蓝绿）；haze = 头灯/相机附近的光散射 -->
  <rgba fog="0.05 0.20 0.24 1" haze="0.10 0.30 0.34 1"/>
  <!-- headlight 是相机自带的光源，水下正好当 ROV 探照灯；active 可随机关，模拟"关灯巡检" -->
  <headlight ambient="0.15 0.16 0.14" diffuse="0.55 0.58 0.55" specular="0.05 0.05 0.05"/>
  <quality shadowsize="2048"/>
</visual>

<!-- 额外光源：聚光 + 衰减，模拟第二盏探照灯 / 工具灯 -->
<light name="flood" pos="1.0 -0.6 1.2" dir="-0.3 0.5 -1" mode="targetbody" target="object"
       cutoff="35" exponent="8" attenuation="1 0 0.3"
       diffuse="0.8 0.82 0.75" specular="0.15 0.15 0.15" castshadow="true"/>
```

可用旋钮（原生）：

| 参数 | 位置 | 物理含义 / 水下对应 |
|---|---|---|
| `fogstart` / `fogend` | `<visual><map>` | 雾的起止距离（× `stat.extent`）→ 水的**能见度** |
| `rgba.fog` | `<visual><rgba>` | 雾色 → 水的**背景色 / 无限远色** |
| `haze`（`map.haze` 与 `rgba.haze`） | `<visual>` | 头灯附近的光散射比例 → 水下**背向散射**（灯亮 + 颗粒多时前面浮起噪声幕） |
| `headlight.ambient/diffuse/specular/active` | `<visual><headlight>` | 探照灯强度与开关 |
| `<light>` 的 `cutoff`/`exponent`/`attenuation` | `<worldbody>` | 聚光锥角、衰减、热点 |
| `statistic.extent`（只读） | Python `env.model.stat.extent` | 上面所有归一化参数的基准，**先打印它** |
| `camera fovy` / `pos` | `<camera>` | 视场角与机位（贴脸 vs 远观，「自遮挡」程度不同） |

**原生做不到、必须放到图像域的**：
- 分通道衰减（红光先消失）——MuJoCo 的雾是统一的，不知道波长
- 距离相关的对比度损失与散射模糊（forward scatter）
- 传感器层面：低照度噪声、白平衡漂移、压缩伪影、暗角

> ⚠️ **2026-09 实测更正（mujoco 3.8.0）**：`model.vis.map.fogstart/fogend/haze` 与
> `rgba.fog/haze` 虽然还能赋值，但 classic renderer **已经不读它们** ——
> `MjvScene` 里没有 fog 字段，`mjRND_FOG` / `mjRND_HAZE` 打开也**没有任何像素变化**。
> 也就是说上面那段 `<visual>` 里的雾参数在 3.8 上是**装饰**：真正生效的只有
> `headlight` 与 `<light>`。所以 L1 的实现策略定为：**灯光交给 MuJoCo，水体光学全部走图像域**。
> 深度渲染则确认可用：`Renderer.enable_depth_rendering()` 返回米制深度，与 RGB 逐像素对齐。

### 2.3 图像域成像模型（推荐做法）

VLA 吃的是 2D 图像，所以**在图像域做水的域随机化，自由度最大、成本最低**。
采用水下成像的标准模型（Jaffe–McGlamey 简化式）：

```
I_c(x) = J_c(x) · exp(-β_c · d(x))  +  B_c · (1 - exp(-β_c · d(x)))
         └── 直接透射：衰减 + 偏色        └── 背向散射：悬浮颗粒被照亮

再叠：
  forward scatter  : 随距离增大的高斯模糊（σ ∝ d）
  sensor           : 低照度高斯/泊松噪声、暗角(vignetting)、JPEG 伪影
  white balance    : 全局色温漂移（模拟自动白平衡在蓝绿环境下失准）
```

- `d(x)`：深度图。`mujoco.Renderer` 可以额外渲染 depth（`renderer.enable_depth_rendering()`），
  或对固定机位用「相机到目标距离」近似（更省，先做近似版）。
- 通道衰减系数典型关系 `β_R > β_G > β_B`（红光先没），清水 / 沿岸 / 浑浊三档预设。
- 全部参数 **3~5 个** → 随机化空间极大且便宜。

**工程铁律**：
1. 同一个函数必须**同时**挂在 `collect_data_vla.py`（写盘前）和 `eval_vla.py`（推理前），
   训练/评测的成像必须一致，否则评测数字无意义。
2. 参数写进 configs，且**把本次采样的参数与 seed 记进 dataset meta**，否则消融实验不可复现。

**✅ 已实现**：`WYSD/underwater_vision/`（本仓库根目录；模组 README 见其 `README.md`）

```python
from underwater_vision import WaterRenderer          # 两行接入
wr = WaterRenderer(model, 224, 224, params="turbid", seed=0)
img = wr.render(data, camera="grasp_cam")            # 输出即水下图
clean, wet = wr.render_pair(data, "grasp_cam")       # 清水/水下成对（消融用）
```

- 参数端点：`WaterParams.preset("coastal")` / `WaterParams.from_visibility(1.5)` /
  `WaterParams.sample(rng, level="medium")`；可 `to_json/from_json/hash`（保证两端一致）。
- 灯光端点：`SceneStyle.preset("deep")`、`WaterRenderer.set_style()`、
  `randomize_scene_lights()`；XML 片段用 `xml_visual_snippet()` / `xml_floodlight_snippet()`。
- 自检：`python -m underwater_vision.selftest`（退出码 0/1，33 项全绿）、
  `python -m underwater_vision.demo`（渲染对照组图到 `underwater_vision/out/`）。
- 报告素材：`python -m underwater_vision.animate --all` 出三张**多视角带注解动画**
  （`task` 三视角抓取-放置 / `water` 四档水况消融 / `light` 四档光照），每张同时出
  GIF + MP4 + `_still.png`。运动源是**真实专家轨迹**（采一次 qpos 序列后只做正运动学，三张图运动逐位一致）。详见 §7 里程碑与模组 README。
- 实测退化（UranUS 抓取场景，相对清水图对比度）：clear 0.81 / coastal 0.56 / turbid 0.28 /
  harbor 0.22 / vis=0.8m 0.16 / 关灯 0.09；红通道塌得远快于蓝通道。

### 2.4 L2 域随机化清单（建议表）

| 维度 | 参数 | 建议范围（起点，需按专家成功率再调） | 备注 |
|---|---|---|---|
| 能见度 | `fogend`（× extent）/ `β_c` | 3 档：清（≈无雾）/ 中（雾起 1m 外）/ 浊（雾起 0.5m） | **最关键**，先做 3 档离散而不是连续 |
| 水体颜色 | `rgba.fog`、`B_c` | 蓝绿 / 青 / 绿褐（沿岸） | `B_c` 应略亮于 `fog` 色 |
| 背向散射 | `haze`、颗粒密度 | 0.0 / 0.3 / 0.6 | 灯亮时最明显 |
| 照度 | `headlight.active`、`ambient` | 开灯 / 关灯（0.05 环境光） | "关灯"档考低光鲁棒性 |
| 光照方向 | `<light>` 的 pos/dir | ±30° 抖动 | 避免策略记住固定阴影 |
| 颜色衰减 | `β_R/β_G/β_B` | 清水 (0.35,0.12,0.08) / 浑浊 ×3 | 使黄色/红色物体变灰 |
| 传感器 | 噪声 σ、JPEG 质量、暗角 | σ ∈ [0,8]/255、q ∈ [40,95] | 最便宜、最有效的一项 |
| 相机 | 外参微抖 ±2cm/±3°、`fovy` ±5° | | 防过拟合固定机位 |
| 目标外观 | 颜色/尺寸/纹理/反照率 | 见 3.3 S2 | |
| 深度估计 | 用近似 d 还是真 depth | 先近似 | 影响真实感但不影响管线 |

### 2.5 水下视觉的难点排序（别搞错重点）

1. **低对比度 / 低纹理** —— 灰色管道、灰色阀盘、沉积物背景，特征极弱（比"蓝绿色滤镜"难得多）
2. **背向散射** —— 开灯后颗粒变噪声幕，越靠越糟
3. **颜色衰减** —— 现有 **黄色方块正是受害者**（黄→灰），这本身就是天然消融实验
4. **自遮挡** —— 机械臂/工具占掉画面一大块，工作区常被自己挡住（腕部相机尤其）
5. **光照不均** —— 探照灯热点 + 边缘黑区，曝光/白平衡漂移

**明确不做**：MuJoCo 里的体积光、焦散、折射。水的折射率会让相机标定、几何、IK 全部错位；
若确需色散/畸变，只在图像域叠加。

---

## 3. 任务设计

### 3.1 水下机械臂任务族（任务 → 难点 → 可自动判吗）

| 族 | 具体任务 | 水下特点 / 难点 | 自动判据 |
|---|---|---|---|
| **A. 科学 / 生物采样** | 轻取珊瑚·海绵·管虫；推芯采样（垂直插管下压）；岩石/结核抓取；温度·流体探针插入 | 易碎（硬抓 = 失败，可用夹持力阈值建模"破损"）；不规则物体夹持面小；细柔顺物件对准精度高 | ✅ 破损用接触力阈值；采样用"入管深度 + 保持时长" |
| **B. 工业干预（最"值钱"）** | 阀手轮旋转（quarter/multi-turn）；**hot stab 液压接头插拔**；扭矩工具套筒对接 + 旋转；连接器插拔；阳极更换；除生物附着刷洗 / 水射流；水下切割；CP/NDT 探针点测 | 阀：旋转 + 力矩保持；hot stab：**mm 级 peg-in-hole + 保持 + 拔出**，最经典也最难；清洁：工具末端沿表面扫掠而非抓取 | ✅ 基本都是阈值 + 持续判定（角度、接触点、保持时长） |
| **C. 回收 / 清理（养殖·环保·考古）** | 死鱼/垃圾/幽灵网回收（拖拽而非抓取）；网箱清网；投饵器·传感器挂取；文物轻取与软刷清理 | 目标柔软、形态变化大，"抓"的语义定义模糊；考古要求"零破损" | ⚠️ 拖拽类需定义"移出区域"判据 |
| **D. 感知 / 巡检（不抓东西）** | 语言指称 + 靠近（"去那个阀门"）；相机贴近巡检（给成像质量判据）；管道/缆线跟随；低能见度下再定位 | 直接考"语言 → 目标定位"的正确性；水下检查是真实刚需、工作量大 | ✅ 距离 / 视角 / 线跟随偏差 |
| **E. 长程组合（VLA 真正拉开差距处）** | 取工具 → 到工作点 → 操作 → 放回；"开阀 → 插探针 → 记录 → 关阀"；同场景不同指令做不同操作 | 考长程记忆、子目标切换、**语言是否真被使用** | ✅ 分阶段里程碑 |

### 3.2 任务规格模板（每新增一个任务，必须填满这张表）

```
任务名        :
族            : A / B / C / D / E
场景 XML      : asset/scene_uranus_*.xml（新增或复用）
被操作对象    : 名字 / geom / 尺寸 / 质量 / 是否 freejoint
语言指令池    : ≥3 个改写模板（见第 4 节），含 1 个 hold-out
观测          : observation.state 维度与含义；用哪几路相机；是否给 privileged 版本
动作          : action 维度与语义（默认沿用"未来 lookahead 拍的实测状态"）
专家          : 反应式里程碑状态机（见 3.5），阶段列表
成功判据      : env.success() 的几行判据（见 3.4）
失败模式      : 本任务特有的失败（滑落 / 破损 / 扬尘 / 缠绕 / 超时）
难度旋钮      : 能见度、初始位姿范围、容差、控制频率
数据量预算    : 目标段数 / 帧数
评测指标      : 成功率 + 附加量（角度误差 / 插入深度 / 保持时长）
```

### 3.3 课程设计 S0 → S6

| 阶段 | 任务 | 语言指令示例 | 成功判据 | 新增工程量 | 验证什么 |
|---|---|---|---|---|---|
| **S0**（已有） | 清水 reach / pick-place | `put the yellow cube on the blue disc` | 抬升 > 2cm 且 `env.success()` | — | 基线（reach 100%，pick-place 策略 5%） |
| **S1** | **同任务 + 水下成像**（只改渲染，任务不动） | 同上 | 同上 | 小：`src/water_image.py` + `<visual>` 块 + 两处调用 | **消融：视觉退化让成功率掉多少** —— 水下鲁棒性的第一张图 |
| **S2** | 换 L0 场景与目标：海床 + 弱纹理 + 灰色目标 | `grab the gray valve handle` / `pick up the rock` | 距离 + 抓持力 | 中：场景 XML + 目标采样 | 低对比度下的视觉定位 |
| **S3** | 两个"判据二值、状态机好写"的工业任务：**阀门旋转 ≥90° 且保持**、**探针点测（接触指定点并保持 2s）** | `turn the valve clockwise` / `place the probe on the marked point` | 角度 / 接触点 + 保持时长 | 中：场景 + env + 专家 | 接触丰富任务的 BC 可行性（比 pick-place 简单且更有代表性） |
| **S3.5** | 加**泥沙扬尘**：动作让沉积物扬起，数秒内什么都看不见 | 同 S3 | 同上 | 小：粒子 + 能见度状态 | "动作历史影响可观测量" → 考记忆与谨慎性（POMDP） |
| **S4** | **hot stab 插入**（mm 级 peg-in-hole + 柔顺/搜索） | `insert the hot stab into the panel port` | 插入深度 + 保持 | 大：模型 + 力/接触感知 + 搜索策略 | 高精度装配 —— 真正的硬骨头 |
| **S5** | **多任务混训 + 语言条件**：S2/S3/S4 共用一个模型，指令 hold-out | 每任务 3~5 个改写模板 | 各任务判据 | 中：数据组织与 meta | **语言分支是否真被用上**（第 6 节） |
| **S6** | 长程：取工具 → 到工作点 → 操作 → 归位 | `pick up the torque tool and open the valve` | 分阶段里程碑 | 大：工具模型 + 长程专家 | 长程规划 + 工具使用 |

**优先级建议**：S1（最便宜、立刻出对比图）→ S3（性价比最高的新任务）→ S5（验证 VLA 本身是否成立）→ S4（难，但论文/项目亮点）→ S2/S3.5/S6。

### 3.4 判据伪代码（放在 `env.success()` 里，沿用现有模式）

阀门旋转（S3）：

```python
def success_valve(self, min_angle_deg=90.0, hold_s=1.0) -> bool:
    """手轮绕其轴累计转过 min_angle_deg，且最后 hold_s 秒内保持不再回弹。"""
    ang = self.valve_angle_deg()                 # 由手轮 body 的四元数解出绕轴角
    return ang >= min_angle_deg and self._held(ang, tol_deg=8.0, hold_s=hold_s)
```

探针点测（S3）：

```python
def success_probe(self, target_site, tol=0.01, hold_s=2.0) -> bool:
    """探针尖端距目标点 < tol，并持续 hold_s；同时接触力 < 阈值（不能压坏）。"""
    d = np.linalg.norm(self.probe_tip() - self.site_pos(target_site))
    return d < tol and self._contact_held(self.probe_geom, target_site, hold_s) \
           and self._max_contact_force() < FORCE_LIMIT
```

插入（S4）：

```python
def success_insert(self, depth_min=0.03, tol_xy=0.003) -> bool:
    """插入深度达标，横向偏移在容差内，且已保持。"""
    return self.insert_depth() >= depth_min and self.insert_xy_err() < tol_xy \
           and self._held(self.insert_depth(), tol=0.005, hold_s=0.5)
```

扬尘（S3.5，是**难度机制**而不是判据）：

```python
# 每次与底质发生大接触 -> 给 visibility 打一个"浑浊脉冲"，按时间常数恢复
if self._sediment_contact_impulse() > THRESH:
    self.turbidity = min(1.0, self.turbidity + 0.5)
self.turbidity *= np.exp(-dt / TAU)            # TAU ~ 2~5s
# 把 self.turbidity 喂给 water_image.py 的 β / B_c / haze
```

### 3.5 专家必须是"反应式"的（本仓库已用血换来的教训）

`README.md` 已记录：pick-place 的专家是**时间驱动的轨迹**，同一状态在不同时刻对应不同动作
→ 任务对策略不可观测 → BC 只能学条件均值 → 学出来的策略 5%。所有新任务必须避免重犯：

- 专家写成 **反应式里程碑控制器**：`看状态决定当前里程碑`（齿面离物块还远 → 去上方；
  已在上方 → 下探；物块在齿间且夹爪开着 → 合爪…），每个状态对应唯一动作。
- 动作语义沿用现有约定（`src/expert_grasp.py` 文件头）：
  `action` = **未来 `lookahead`(默认 4，0.2s) 拍的实测关节角**（不是"当前目标"，否则退化成恒等映射 f(s)≈s，闭环里策略停在半空不动）；
  夹爪用**下发命令**而不是实测角（夹住时实测角被顶住，用实测角训出来的策略给不出夹持力）。
- 新增任务直接复用 `build_frames(stream, lookahead)` 与 `env.move_to / move / release` 那套底座，
  这样 `collect_data*.py` / `train.py` / `eval*.py` 都不用改维度适配。

### 3.6 每个任务都要出一个 `env.success()`

不能几行代码自动判成功的任务先别做。
现有 `UranusGraspEnv.success(pos_tol=0.05, gripper_open_tol=0.1)` 就是这个模式的样板。

---

## 4. 语言指令设计（模板池 + hold-out）

现在指令是硬编码的一句。多任务/语言泛化需要：

| 维度 | 变体示例 |
|---|---|
| 动词 | `put` / `place` / `move` / `drop` |
| 指称方式 | 颜色（`yellow cube`）、位置（`cube on the left`）、功能（`the valve handle`）、混合（`the gray handle near the pipe`） |
| 句式 | 祈使句 / 简短名词短语 / 带冗余修饰 |
| 干扰项 | 场景里放 2 个同类目标，靠语言区分（**考语言真的被用上**） |
| 方向词 | `clockwise` / `counter-clockwise`、`in` / `out`、`upper` / `lower` |

**做法**：
- 建 `configs/instructions.json`：每个任务一组模板（≥3 句）+ 一组 **hold-out 句**（不参与训练，只评测）。
- 采集时**逐段随机**从模板池抽一句写进 `task` 字段（同一场景不同指令 → 数据里天然含"语言-行为"配对）。
- 评测分两组：**seen 指令**（训练见过的模板）与 **unseen 指令**（hold-out）。
  unseen 掉太多 = 语言泛化不行（正常现象，但要量化）。

---

## 5. 数据方案

### 5.1 格式与组织（沿用 LeRobotDataset）

```
data/underwater/
├── s1_imaging_grasp/     # S1：清水 vs 水下两套（同一场景、同一专家，唯一变量是成像）
├── s2_lowcontrast/       # S2：海床 + 弱纹理 + 灰色目标
├── s3_valve/
├── s3_probe/
├── s4_hotstab/
└── meta/                 # 每个 dataset 的环境参数与 seed 记录
```

约定不变：
- `observation.images.top` / `.wrist`（+ 可选 `front`/`tool`）、`observation.state`(7 或按任务扩展)、
  `action`、`task`(str)，fps=20。
- **状态里不给被操作物位姿**（`--privileged_state` 只做对照），水下尤其要对，否则策略走捷径、永远学不会"在雾里看图"。
- ⚠️ `LeRobotDataset.create()` **要求目录不存在、不能续采**（README 已记录）。多任务要么一次性采完，
  要么每任务一个 dataset 再合并；采集时留足时间（现有 300 段耗时 27 分钟，加了图像后处理会更慢）。
- 图像后处理放在写盘前 → 体积会涨（现有 300 段 / 50700 帧 / 1.1GB），必要时降到 224² 或改用视频编码。

### 5.2 数据量预算

| 场景 | 建议量 | 理由 |
|---|---|---|
| 单任务（S1~S4） | ≥ 100~300 段 / ≥ 3~5 万帧 | 对齐现有 pick-place VLA 基线（300 段 / 50700 帧） |
| 多任务（S5） | **每任务 ≥ 50~100 段** | VLA 数据饥饿：1 任务 300 段 ≫ 10 任务各 30 段 |
| 消融对照（S1） | 清/浊两套各 ≥100 段，**同一 seed 序列** | 保证唯一变量是成像 |

**纪律**：不要为了"任务多"而摊薄每任务数据量。宁可 3 个任务各 100 段。

---

## 6. 评测矩阵（这决定项目的说服力）

### 6.1 三类实验

| # | 实验 | 做法 | 产出 |
|---|---|---|---|
| 1 | **水下 vs 清水（S1 消融）** | 同一策略在两种成像下评测 | 成功率差值 = 视觉退化代价 |
| 2 | **能见度 × 光照参数扫描** | 固定策略，扫 `fogend`/`β`/`headlight.active` 网格 | **成功率–能见度曲线**（比单个数字有说服力得多，适合当主图） |
| 3 | **语言消融（必须有）** | 同场景同初始状态，只换指令（`clockwise` ↔ `counter-clockwise`；指称换目标） | 动作分布是否改变 / 是否指向新目标 → **语言分支是否被用上** |
| 4 | privileged 对照 | `--privileged_state` 版本同数据训练 | 图像分支是否真在贡献（对照 S1） |
| 5 | 相机消融 | `--images top` / `wrist` / `top,wrist` | 哪一路在水下更有用（腕部近 → 背向散射更重） |

### 6.2 指标定义（每个任务都应报）

- **主指标**：成功率（与现有 `eval_vla.py` 判据一致）。
- **过程量**：抬升高度、放置/对准误差（现已有 err 输出）、阀门角度误差、插入深度、保持时长、接触力峰值。
- **鲁棒性**：成功率对能见度的斜率（"每档能见度掉多少"）。
- **泛化**：unseen 指令成功率、未见初始位姿范围成功率。

### 6.3 评测脚本要求

在 `eval_vla.py` 基础上加：
1. `--instruction`（覆盖默认指令）→ 支持语言消融；
2. `--water_preset {clear,turbid,...}` + `--water_params` → 支持参数扫描；
3. 结果**输出结构化（JSON/CSV）**，方便画曲线，而不是只打印一行成功率；
4. 训练与评测调用**同一个** `water_image.py` 函数（铁律）。

---

## 7. 工程落地清单（文件级）

| # | 文件 | 动作 | 说明 |
|---|---|---|---|
| 1 | `underwater_vision/`（`params/image/scene/render/demo/selftest/preview/animate`） | ✅ **已完成** | 成像模型 + 预设/能见度换算/域随机化 + 灯光风格 + `WaterRenderer` 封装 + 自检（33 项全绿）+ 终端看图 + **报告动画生成器**（多视角带注解 GIF/MP4）。**已并入本仓库根目录 `WYSD/underwater_vision/`**，`PYTHONPATH=E:\projects\mujoco\WYSD` 即可 import |
| 2 | ~~`configs/water.json`~~ | ✅ 并入模组 | 参数随 dataset 存 `WaterParams.to_json()`（`<dataset>_water.json`），不再单独维护 configs |
| 3 | `asset/scene_uranus_underwater_grasp.xml` | **新增**（未做） | L0 海床场景 + 探照灯 `<light>`；`<visual>` 块可直接用 `underwater_vision.xml_visual_snippet()` 生成（**雾参数在 mujoco≥3.8 无效**，别指望它） |
| 4 | `collect_data_vla.py` | 改（未做） | 用 `WaterRenderer` 替换 `mujoco.Renderer`；`grab_images()` 里两行变 `renderer.render(env.data, camera=cam)` 一行；把 `params.to_json()` 写进 dataset 旁边 |
| 5 | `eval_vla.py` | 改（未做） | `ObsBuilder` 里 `WaterRenderer.wrap(self.renderer, params=WaterParams.from_json(...))`；加 `--instruction` / `--water_json`；结果写 JSON |
| 6 | `src/env_grasp.py` 或新 `src/env_underwater.py` | 新增/扩展 | 新任务的 `success()` / 目标采样 / 扬尘状态 |
| 7 | `src/expert_valve.py` / `expert_probe.py` / `expert_hotstab.py` | **新增** | 反应式里程碑专家（复用 `expert_grasp.py` 的 `build_frames` 与 `move_to`） |
| 8 | `collect_data_underwater.py` | **新增** | 多任务采集入口（任务名 → 场景/env/专家/指令池 分派） |
| 9 | `configs/instructions.json` | **新增** | 指令模板池 + hold-out（第 4 节） |
| 10 | `grasp_e2e.py` 同构的 `*_e2e.py` | **新增** | 每个新任务的专家随机布局回归（要求 100%），专家不过关就不许采数据 |
| 11 | `README.md` / 本文档 | 改 | 记录新任务现状、数据清单、坑 |

**沿用而不改的**：`action`/`obs` 语义、`lookahead`、`MIN_LIFT`、`train.py`、`eval_grasp.py`
的维度自适应逻辑 —— 新任务尽量对齐 7 维 action，能省掉大量适配工作。

---

## 8. 风险与坑

### 8.1 "不考虑水动力" ≠ "不考虑水的其他效应"

中性浮力、末端水阻、液压低频带宽，都会让**同一个任务在水下变难**。不改动力学也能近似：

- 关节加阻尼、末端加负载；
- 控制频率从 20Hz 降到 5~10Hz（对应液压/推进器带宽）；
- 动作加 1~2 拍延迟。

**必须在文档与代码注释里注明这是"任务难度旋钮"，不是物理仿真**，免得被当成水动力建模。

### 8.2 其他坑（按踩到的概率排序）

| 坑 | 后果 | 对策 |
|---|---|---|
| 随机化范围超过专家能力 | 采到的数据几乎全是失败样本 → 策略学会"放弃" | 先标定"专家在哪些参数区间内仍 100%"，把随机化限制在区间内（结合 `*_e2e.py` 回归） |
| 成像只在采集端做、评测端忘了做 | 评测数字虚高/虚低，实验全废 | 铁律：同一函数两处调用，代码里加断言（参数哈希写入 meta 与 ckpt 旁） |
| 扬尘变成 POMDP | 单帧 BC 学不会（同一观测对应不同最优动作） | 要么先把扬尘关掉；要么给观测加历史帧 / 用带记忆的策略 |
| 只加青绿色滤镜 | 实验价值≈0 | 至少包含：分通道衰减 + 背向散射 + 噪声三项 |
| 在 MuJoCo 里做折射/焦散 | 相机标定与几何全错 | 只在图像域做色散/畸变 |
| 多任务摊薄数据 | 每个任务都不收敛 | 每任务 ≥50~100 段 |
| 任务太多、判据写不出来 | 无法自动评测 | 先过"3.4 判据伪代码"这一关再开工 |
| 忘了 `stat.extent` 重标定 | 雾参数错位，画面全白/无雾 | 先打印 `env.model.stat.extent` 再调参 |
| 8GB 显存硬上 π0 | 直接 OOM | SmolVLA(450M) 起步（`freeze_vision_encoder` / `train_expert_only`），π0 上云 A100 LoRA（见 `VLA_FINETUNE.md`） |

---

## 9. 里程碑与优先级

| 里程碑 | 内容 | 判定标准（可验收） |
|---|---|---|
| **M1** | S1：水下成像 + 消融 | 清/浊两套数据各 ≥100 段；同一策略在两套成像下的成功率曲线（3 档能见度） |
| **M2** | 一个工业任务（阀门旋转或探针点测） | 专家 `*_e2e.py` 随机 20 组 = 100%；`env.success()` 可自动判；≥100 段数据 |
| **M3** | SmolVLA 微调出第一版 | 成功率 > 基线 MLP（pick-place 5%）；有 unseen 指令评测 |
| **M4** | S5 多任务 + 语言消融 | 单模型跑 ≥3 任务；指令替换实验证明动作随语言改变 |
| **M5** | S4 hot stab | 插入成功率 + 容差报告（mm 级） |
| **M6**（可选） | sim→real 视觉域对齐 | 用真实水下图像/视频拟合 β 与 B_c，缩小域差 |

---

## 10. 外部线索（粗检索，需自行核实）

- 水下 VLA 方向的项目线索：[USIM / U0 项目页](https://vincentgu2000.github.io/u0project/)
- "无水下遥操作"的水下操作学习（思路与"用已有 VLA 微调"高度接近）：
  [UMI-Underwater](https://browse-export.arxiv.org/pdf/2603.27012)
- 水下成像模型：Jaffe–McGlamey 简化式（第 2.3 节），是水下视觉复原/仿真的通用起点。

> 以上链接来自一次粗检索，未逐篇精读；引用前请自行核对内容与结论。