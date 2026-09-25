# Lerobot-Uranus-VLA-Demo

把 LeRobot-MuJoCo-VLA-Tutorial（本仓库开发时的同级目录 `../Lerobot-MujoCo-VLA-Tutorial`）
的 VLA 训练流程"移植"到一个不同的机械臂 —— **Uranus 液压大臂** —— 的完整 demo：
**环境 → 专家 → 数据（含人手遥操作）→ 训练 → 评测 → VLA 微调准备**。

Uranus 与教程里的 Open Manipulator Y（桌面小臂）差异很大：

| 维度 | 教程 (OM-Y) | 本 demo (Uranus) |
|------|-------------|------------------|
| 尺寸 / 工作半径 | 桌面小臂，~0.5 m | 大臂，TCP 半径 ~1.2–2.0 m |
| 自由度 | 6 关节 + 4 夹爪关节 | 6 关节 + 2 个直驱夹爪关节 |
| 传动 | 串联电机直驱 | 真机 joint2/3 为液压缸并联（本 demo 模型里已简化为串联位置伺服，见下） |
| 关节空间 vs 执行器空间 | 相同（8 个执行器 = 8 关节） | 本 demo 模型同样是 8 执行器 = 8 关节；真机是 6 关节 ↔ 7 ctrl，需 `CylinderMapper` |
| 任务 | 抓取 + 放置 + 开关柜门 | **reach（点到点到达）已跑通**；**方块抓取-放置**环境+专家+数据管线已就绪，学出来的策略还不行（见下） |

## 三条链路各自到哪一步了

| 链路 | 状态 |
|---|---|
| **reach**（点到点到达） | ✅ 全通：IK 专家采数据 → MLP 行为克隆 → 评测 **100/100** |
| **抓取-放置**（轻量 npz + MLP） | ⚠️ 环境/专家/采集/训练/评测全部就绪，专家 **100%**；**学出来的策略只有 5%**（原因与后续方案见"仓库导览"一节） |
| **抓取-放置 VLA**（图像+语言，π0/SmolVLA） | ✅ 数据管线就绪（LeRobotDataset 300 段 / 50700 帧）；⚠️ **微调受本机显存限制**，见 `VLA_FINETUNE.md` |

---


## 目录结构

```
Lerobot-Uranus-VLA-Demo/
├── asset/
│   ├── scene_uranus.xml                 # reach 场景：地面 + 目标点 + 机械臂 include
│   ├── scene_uranus_grasp.xml           # 抓取场景：桌子 + 方块 + 放置点 + 相机 + 离屏渲染设置
│   └── uranus/
│       └── uranus_arm_gripper_model.xml # Uranus 简化模型（关节限位 + tcp_link + 齿面 geom）
├── uranus/
│   └── control/
│       ├── __init__.py
│       └── cylinder_mapper.py           # 关节角 -> 液压缸 ctrl 映射（历史参考，真机用）
├── src/
│   ├── env.py                           # UranusReachEnv（原生 mujoco，可无头运行）
│   ├── env_grasp.py                     # UranusGraspEnv：抓取/放置 + 齿面 IK + 夹爪标定
│   └── expert_grasp.py                  # 抓取专家状态机 + 录制约定（动作/观测语义）
├── configs/task.json                    # reach 任务配置
├── collect_data.py / train.py / eval.py # reach 三件套
├── collect_data_grasp.py                # 抓取：脚本专家采轻量 npz
├── collect_data_vla.py                  # 抓取：脚本专家采 LeRobotDataset（图像+语言）
├── manual_collect.py / run_manual.ps1   # 抓取：人手键盘遥操作采集（tkinter 界面）
├── eval_grasp.py                        # 抓取：MLP 策略闭环评测
├── eval_vla.py                          # 抓取：lerobot 策略（pi0/SmolVLA）闭环评测
├── grasp_e2e.py                         # 抓取专家验证：python grasp_e2e.py --randomize 20
├── test_env.py                          # reach 环境冒烟测试
├── debug/                               # 2026-09-11 那轮联调的 38 个脚本（索引见 debug/README.md）
├── docs/                                # 验证截图
├── VLA_FINETUNE.md                      # π0 / SmolVLA 硬件门槛、环境、微调命令
└── requirements.txt                     # 依赖分三档：核心 / 遥操作 / VLA
```

常用入口：

```bash
python test_env.py                        # reach 环境冒烟（ALL CHECKS PASSED）
python grasp_e2e.py --randomize 20        # 抓取专家随机布局回归（100%）
python run_manual.ps1                     # 人手遥操作采集（Windows）
python collect_data_vla.py --episodes 300 # 生成 VLA 数据集（LeRobotDataset）
python eval_grasp.py --num_episodes 20    # 评测抓取策略
```

## 快速开始

```bash
# 1. 冒烟测试：验证模型、IK、step 是否正常
python test_env.py

# 2. 采集专家示范（IK 生成轨迹）
python collect_data.py --num_episodes 200 --episode_len 100

# 3. 训练行为克隆策略
python train.py --epochs 100

# 4. 评测成功率
python eval.py --num_episodes 50 --verbose
```

> 使用与教程相同的 conda 环境 `lerobot_mujoco_vla_tutorial`（含 mujoco 3.2.1、
> torch 2.7.1、numpy）即可运行，本 demo 不依赖 lerobot / transformers / 图像。

---

## 与教程流程的对应关系

| 教程阶段 | 教程文件 | 本 demo 对应 |
|----------|----------|--------------|
| 数据采集（遥操作） | `collect_data.py` / `0.teleop.ipynb` | `collect_data.py`（IK 专家，无遥操作） |
| 数据变换 | `10.transform.ipynb` / `transform.py` | 内嵌在 `train.py` 的归一化 |
| 训练 | `11.train_custom.ipynb` / `train_custom.py` | `train.py`（纯 torch MLP） |
| 评测 | `12.eval_custom.ipynb` | `eval.py`（成功率） |
| 环境 | `src/env/env.py` + `MuJoCoParserClass` | `src/env.py`（原生 mujoco） |

## 数据 / 动作 / 观测维度

```
obs    = [joint1..joint6 (当前, 6), goal_x, goal_y, goal_z (3)]  -> 9 维
action = [joint1..joint6 (目标关节角 = IK(goal), 6), gripper (1, 本任务恒开)] -> 7 维
```

- 控制模式为 `joint`（绝对关节角），与教程 `configs/train.json` 的 `control_mode: joint` 一致。
- **action 语义**：策略学的是 `goal -> 目标关节角`（神经 IK），即每步输出 IK 解，而不是
  "下一帧关节角"。若用"下一帧关节角"做闭环轨迹跟踪，微小预测误差会在闭环中逐步累积
  （behavior cloning 的分布漂移），成功率会明显下降；用绝对 IK 目标则每步都朝正确目标走。
- 夹爪在到达任务中始终打开，`action[-1]` 训练为 0。
  夹爪是两个直驱位置关节（`jaw1.1 / jaw2.1`）：**角度越大越张开**
  （0.22 rad 时齿面净间距约 6cm，-0.07 rad 时两齿面刚好贴住），
  已在 `src/env.py` 的 `GRIPPER_OPEN/CLOSE` 与 `src/env_grasp.py` 的
  `tooth_clearance()/grasp_angle()` 中标定。

---

## Uranus 特有适配点（本 demo 的关键）

### 1. 关节空间 ≠ 执行器空间（真机） / 串联简化（本 demo）

真机上 Uranus 只有 7 个执行器：joint2/joint3 由两个液压缸通过 `<equality connect>`
并联驱动，因此关节角不能直接塞进 `data.ctrl`，必须用
`uranus/control/cylinder_mapper.py` 的余弦定理映射成缸体滑动量：

```
c(q) = sqrt(a^2 + b^2 - 2ab·cos(theta_home + s·(q - q_home)))
ctrl(q) = c(q) - c_home
```

但该并联机构（equality ball 约束 + 极硬的缸体执行器）在 `mj_step` 动力学下数值极不稳定，
无法稳定跟踪关节目标，做接触动力学抓取更是不可能。所以本 demo 的
`asset/uranus/uranus_arm_gripper_model.xml` 把 joint2/joint3 直接改成串联位置执行器
（`cylinder_mapper.py` 保留作真机参考），于是 `data.ctrl = [joint1..joint6, jaw1.1, jaw2.1]`
共 8 维，与关节一一对应；FK/IK / 关节限位 / TCP 都与真机一致。

### 2. 大工作空间

Uranus 的 TCP 工作半径约 1.2–2.0 m，所以目标采样用极坐标：

```json
"goal": {
    "r_range": [1.2, 1.75],
    "theta_deg_range": [-80.0, 80.0],
    "z_range": [0.20, 0.90]
}
```

每个采样目标会用 IK 验证可达（`env.sample_goal()`），不可达则重新采样。

### 3. 关节限位

原始简化模型只有 joint4 带限位，本 demo 在
`asset/uranus/uranus_arm_gripper_model.xml` 中补齐了 6 个臂关节的限位，使 IK 能在
合理范围内求解。

### 4. TCP 坐标系

在 link6 下新增了 `tcp_link` body（教程风格 FK 需要），用于读取末端位置。

---

## 运行结果示例

完整跑通（200 episodes × 100 帧，训练 100 epoch，评测 100 episodes）：

```
Success rate: 100/100 = 100.00%
Final TCP-goal dist: mean=0.016, median=0.011
```

> 该任务较简单（纯关节空间到达 + IK 专家 + 低维 MLP），因此很容易收敛到高成功率；
> 它的价值在于验证"Uranus 关节空间 -> 执行器空间"的适配链路，以及为后续接入图像 /
> 完整 LeRobot 流程打好环境骨架。

---

## 接入完整 LeRobot / 图像 VLA：已经做了（数据），只差微调

本 demo 的 MLP 链路刻意保持轻量（纯 numpy + torch，无 GPU 也能跑）。图像 VLA 那条线已经落地：

1. ✅ `collect_data_vla.py` 用 MuJoCo 离屏渲染出 **两路 RGB**（场景 + 腕部），写成标准
   **LeRobotDataset**（parquet + 图像 + meta），带语言指令；
2. ✅ 观测只给 7 维本体感知、**不给物块位姿**，逼策略真的看图；
3. ✅ `eval_vla.py` 能加载 lerobot 策略在本环境闭环评测（判据与其它脚本统一）；
4. ⚠️ **微调**：π0（3.3B）需要 ≥24GB 显存（官方是 8×A100），本机 8GB 干不了 ——
   要么租云 GPU，要么先用 SmolVLA(450M)。硬件对照、环境安装、具体命令都在
   **[`VLA_FINETUNE.md`](VLA_FINETUNE.md)**。

> 教程目录里的 `uranus_migration_guide.md` 记录了"把教程机器人换成 Uranus 要改哪些硬编码点"，
> 本 demo 就是按那份指南落地的。

---

## 抓取(pick-and-place)任务 —— 2026-09-11 打通

在 reach 之上加了 `src/env_grasp.py` + `asset/scene_uranus_grasp.xml`：
桌上放一个方块，`approach -> 下探 -> 闭爪 -> 抬升 -> 移到放置点 -> 下降 -> 缓开爪`。
端到端验证脚本：`python grasp_e2e.py --randomize 20`（当前 **4cm** 方块：专家 20/20 = 100%，
抬升 139mm、放置水平误差 1.2mm）。

### 场景几何（踩过的坑都写在 XML 注释里）

| 现象 | 原因 | 处理 |
|---|---|---|
| 手臂下不到物块中心（差 6.5cm，`link4_beam ↔ table_top` 接触） | 桌面从 x=0.95 铺到 1.85，而 Uranus 在桌面高度抓取时整条臂几乎水平，肘/前臂正好占住 x≈1.0~1.47、z≈0.29~0.35 | 台面缩小并前移：x∈[1.52,1.82]，抓取点放在 x=1.60 |
| 手臂仍下不去（差 1cm，`link6_mount ↔ table_top`） | 夹爪本体比齿面中点低 3cm，水平姿态下扎进桌面 | 抓取姿态下俯 `GRASP_PITCH=10°`（`solve_ik_grasp` 的姿态目标改成 Ry(10°)），腕部整体抬高 |
| 闭爪时物块被挤出 9cm | 齿面法向随 jaw 角转 2a，张得越开两齿面越不平行 | `align_jaws()`：按物块宽度算出抓取角 a\*，给两片齿各预旋 ∓a\*，夹持时两齿面严格平行 |
| 夹住却抬不起来 / 齿面像"扎"进物块 | 默认 20ms 软接触配 kp=3000 的位置伺服，齿面被压进物块近 2cm，齿的底面反而把物块往桌上按 | 齿面 + 物块 geom 设 `solref="0.002 1"`（接触调硬） |
| 放下时物块被弹飞 17cm | 直接把夹爪 ctrl 从闭合角跳到张开角 | `release()`：按 3s 线性缓慢张开（误差降到 ~1.5cm） |
| 抬升时物块被丢在原地 | `solve_ik_grasp` 内部要改关节角算雅可比，等于把手里夹着的物块留在原地再瞬移手臂 | 该函数默认 `restore_state=True`，只求解不改仿真状态 |
| 重力垂沉导致 IK 到位差 ~1cm | 位置执行器静态误差 | `move_to()` 闭环：用实测齿面中点残差反过来修正 IK 目标，迭代到 mm 级 |

### 物体尺寸

尺寸只写在 `asset/scene_uranus_grasp.xml` 的 `object_geom` 上，
`UranusGraspEnv` 会自动读它标定闭合角、张开角与齿面平行补偿。实测（真实 XML，含重编译）：

| 边长 | 张开净间距 | 结果 |
|---|---|---|
| 3cm | 38.1mm | PASS（误差 5.6mm） |
| **4cm（当前默认）** | 48.1mm | **PASS（4.6mm）**；比 5cm 更容易"不碰到"：齿条 x 方向容差从 ±11mm 放宽到 ±16mm |
| 5cm | 60.5mm | PASS（3.2mm） |
| 6cm | 68.6mm | PASS（0.4mm） |
| 7cm 及以上 | 78.9mm | FAIL —— 超出夹爪能力：张开净间距上限约 9.9cm（jaw 角 0.35），逼近极限时齿面在接近阶段就会把物块碰跑 |

### 数据采集 / 训练 / 评测（管线已搭好，策略还没训出来）

```
src/expert_grasp.py        脚本专家（状态机）：approach->descend->close->lift->transport->place_down->release
grasp_e2e.py               专家验证：python grasp_e2e.py [--randomize N]
collect_data_grasp.py      脚本专家自动采集 -> data/grasp/data.npz
manual_collect.py          手动（键盘遥操作）采集 -> data/manual/data.npz
train.py                   同一个 MLP 行为克隆（维度自适应）-> ckpt/grasp/policy.pt
eval_grasp.py              闭环 rollout + env.success() 成功率
```

| 环节 | 现状 |
|---|---|
| 专家（随机 20~30 组布局） | **100% 成功**，抬升 139mm、放置水平误差 1.2mm（max 1.7mm） |
| 采集（脚本专家，轻量 npz） | 200 段 / 33800 帧，专家成功率 100%，耗时 24s |
| 采集（人手遥操作，轻量 npz） | 6 段 / 3453 帧（"在动"的帧占 53%，脚本专家只有 28%） |
| 采集（VLA 图像格式） | 300 段 / 50700 帧 / 1.1GB，两路 224² 图像 + 语言，耗时 27 分钟 |
| 数据格式 | obs 17 维、action 7 维，npz 三件套（observations / actions / episode_ends） |
| 评测管线 | **用专家录下来的动作回放，能复现任务**（抬升 15cm、误差 2mm）→ 管线本身是忠实的 |
| 学出来的策略 | **尚未成功**（0~5%）：能开到物块上方 2~3cm 并把夹爪合上，但抓不稳/抬不起来/放不到位 |

**obs（17 维）**：`[关节(6), 物块位置-齿面中点(3), 物块四元数(4), 放置点-齿面中点(3), 夹爪(1)]`
—— 位置量都用相对齿面中点的量，绝对坐标版本泛化明显更差。

**action（7 维）**：`[未来 K 拍(默认0.1s)后的实测关节角(6), 未来 K 拍后下发的夹爪命令(1)]`。
这两处都不是随手定的，是踩出来的（细节见 `src/expert_grasp.py` 文件头）：

1. 位置伺服把目标跟得极紧（97% 的帧 |目标-当前| < 0.01 rad），单步学"当前目标"会退化成
   近似恒等映射 f(s)≈s → 闭环里任何状态都是不动点，实测策略停在半空不动；
   改成预测"K 拍后的实测关节角"，动作才系统地领先当前状态。
2. 夹爪必须用"下发命令"而不是"实测角"：夹住物块时实测角被顶在 ≈0.153、到不了命令值
   0.135，用实测角训出来的策略给不出夹持力，抬起来就滑掉（这一条把成功率从 0% 拉到 5%）。

**下一步（按性价比排序）**：

1. **手动遥操作采集**（已做，见下一节）：人手是反应式的，同一个状态对应同一个动作，
   比时间驱动的脚本专家更适合当 BC 的老师；而且流程和教程的 `0.teleop.ipynb` 对齐。
2. **把专家也改成"反应式"的里程碑控制器**：现在的专家是时间驱动的轨迹，同一个状态在不同
   时刻对应不同动作（任务不可观测），策略只能学条件均值。改成"看状态决定当前里程碑"
   （齿面离物块还远→去上方；已在上方→下探；物块在齿间且夹爪开着→合爪…）后，每个状态
   对应唯一动作，BC 才对得上，也才能接着做 DAgger。
3. **DAgger**：用策略自己跑出来的状态去问反应式专家该怎么做，补进数据集，专治闭环漂移。
4. **更大的数据量 + 时序/分块策略**（ACT 式 action chunking 或 diffusion policy）：
   单帧 MLP 对这个接触丰富的任务容量不够。
5. 或者先把任务放宽：加大容差（比如放到浅盒子里、物块再大一点），让"能跑通"先于"做得精"。

### 手动（遥操作）采集：`manual_collect.py`

对应教程的 `0.teleop.ipynb` / `collect_data.py`（人用 leader 臂遥操作），这里改成键盘直接
驱动仿真里的齿面中点：

```
W/S   齿面中点沿 +x / -x（朝 / 离机器人）      Q/E   沿 +z / -z（升 / 降）
A/D   沿 +y / -y（左 / 右）                    Space 夹爪开 / 合
[ ]   调慢 / 调快速度                          R     换一组随机布局（丢弃当前录制）
Enter 结束本段并存盘                            Backspace 放弃本段        Esc 退出
```

```bash
python manual_collect.py                       # 开窗口，键盘遥操作
python manual_collect.py --speed 0.08          # 慢一点更好对位
python manual_collect.py --selftest --episodes 20   # 不开窗口：脚本化"假人"跑一遍，验证录制/存盘
```

- **操作手感与专家一致**：每拍把齿面中点目标做几步齿面中点 IK，再下发给 8 个位置执行器，
  跑一个控制周期（默认 50ms）；夹爪用的是同一套按物块宽度标定的夹持角，所以"合爪"是真的夹住。
- **数据格式与专家完全一致**（`obs` 17 维、`action` 7 维、同一套 `lookahead` 约定，
  见 `src/expert_grasp.py` 的 `build_frames`），所以 `train.py` / `eval_grasp.py` 不用改，
  手动数据和专家数据还可以直接混在一起训。
- **已验证**：`--selftest` 用脚本化的"假人"把目标沿抓取路径推 20 遍；再把录下来的数据回放，
  能复现任务（抬升 133mm、放置水平误差 11mm、`success=True`）——说明遥操作控制、录制、存盘
  这条链路是通的。窗口换成自控 tkinter 之后也实测过：按住 `W` 工具点持续前移、松开即停。
- 真机/真按键那一下没法自动测，逻辑与 `--selftest` 走的是同一条 `control_tick()`。
- **实际录制结果**：6 段 / 3453 帧（`data/manual/data.npz`），"在动"的帧占 53%
  （脚本专家数据只有 28%），人手数据的信息密度明显更高。

### VLA 路线（π0 / SmolVLA）：数据已就绪，微调看硬件

| 文件 | 作用 |
|---|---|
| `collect_data_vla.py` | 脚本专家 → **LeRobotDataset**（两路图像 + 语言 + 状态 + 动作），零人力，几十秒出一批 |
| `eval_vla.py` | 用 lerobot 策略（pi0 / SmolVLA / ACT）在本环境闭环评测；`--dummy_mlp` 可先自检回路 |
| `VLA_FINETUNE.md` | 硬件对照表、环境安装、微调命令、现状与预期 |

已生成 `data/grasp_vla/`：**300 段 / 50700 帧 / 1.1GB**（生成 27 分钟，专家成功率 100%），
fps=20，两路 224² 相机（`grasp_cam` 场景视角 + `wrist_cam` 腕部视角），
语言指令 `put the yellow cube on the blue disc`；
`observation.state` 只给 7 维本体感知（**不含**物块位姿，逼策略真的看图）。

**π0 的硬门槛**：3.3B 参数，LoRA 微调也要 ≥24GB 显存（官方 openpi 配方是 8×A100），
本机 RTX 4060(8GB) 干不了 —— 要么租云 GPU（A100 40G 约 ¥5~10/小时，小数据集 1~2 小时），
要么退一档用 SmolVLA(450M) 或本仓库的 MLP。详见 `VLA_FINETUNE.md`。

## 仓库导览与当前状态（接手时先看这一节）

### 1. 主流程按任务分四条线

| 任务 | 采集 | 训练 | 评测 / 验证 |
|---|---|---|---|
| reach（点对点到达，教程对齐的那条） | `collect_data.py` | `train.py` | `eval.py`、`test_env.py` |
| 抓取-放置（脚本专家，轻量 npz） | `collect_data_grasp.py` | `train.py` | `eval_grasp.py`、`grasp_e2e.py` |
| 抓取-放置（**人手动遥操作**，轻量 npz） | `manual_collect.py`（`run_manual.ps1` 一键启） | `train.py` | `eval_grasp.py` |
| 抓取-放置（**VLA：图像 + 语言**） | `collect_data_vla.py` | 见 `VLA_FINETUNE.md`（lerobot-train） | `eval_vla.py` |

- 抓取环境和专家：`src/env_grasp.py`（环境 + 齿面 IK + 夹爪标定）、`src/expert_grasp.py`（专家状态机 + 录制约定）。
- 场景/模型：`asset/scene_uranus_grasp.xml`、`asset/uranus/uranus_arm_gripper_model.xml`。**几何约束（台面近边必须留在 x=1.52、齿条下俯 10°、夹爪本体 3cm）都写在 XML 注释里，改场景前先读。**
- `debug/` 是 2026-09-11 那轮联调的 38 个调试脚本，**不是主流程**，索引见 `debug/README.md`；`docs/` 放验证用的截图。

### 2. 数据 / 权重清单

| 路径 | 内容 | 状态 |
|---|---|---|
| `data/reach/data.npz` | 5900 帧 / 100 段，9→7 维 | 与 `asset/scene_uranus.xml` 一致 |
| `data/grasp/data.npz` | 200 段 / 33800 帧，17→7 维（脚本专家） | 与当前 4cm 场景一致（2026-09-25 重录） |
| `data/manual/data.npz` | **6 段 / 3453 帧**（人遥操作，"在动"帧占 53%） | 与当前 4cm 场景一致 |
| `data/grasp_vla/` | LeRobotDataset：**300 段 / 50700 帧 / 1.1GB**，两路 224² 图像 + 语言 + 7 维状态/动作，fps=20 | 与当前场景一致（生成 27 分钟，专家 100%） |
| `ckpt/reach/policy.pt` | reach 策略 | 评测 100/100 |
| `ckpt/grasp/policy.pt` | 抓取策略（在 200 段脚本数据上重训） | **1/20 = 5%** |
| `ckpt/manual/policy.pt` | 手动数据策略（6 段） | **0/20 = 0%**（lift 1.2mm，根本没抓到） |

> 换过场景/物块尺寸之后，**旧数据就作废**（观测和动作的分布都变了），要重新采。当前统一在
> **4cm 方块 + 40×44cm 台面 + 抓取姿态下俯 10°** 这一版场景上。

### 3. 怎么接着往下做（按性价比）

1. **攒数据（零人力）**：`python collect_data_vla.py --episodes 300`（脚本专家，几分钟出一批）。
2. **本地试 SmolVLA**：按 `VLA_FINETUNE.md` 建环境 → `lerobot-train`。
3. **上云摸 π0**：A100 40G，LoRA 一两小时；本机 8GB 显存干不了（原因见 `VLA_FINETUNE.md` 的硬件表）。
4. **放宽任务**：更大的盘子/浅盒、方块离盘子更近、判定阈值放宽 —— 让成功率先有正反馈。
5. 想继续手录就 `run_manual.ps1`（默认追加到 `data/manual/data.npz`，不会覆盖旧数据）。

### 与 reach 的关系

reach 链路（`collect_data.py` / `train.py` / `eval.py` / `test_env.py`）不受影响，
仍是 9 维 obs / 7 维 action 的行为克隆；抓取任务的 `env_grasp` 提供 17 维 obs 与
`success()`，采集/训练/评测三件套已经就位（见上）。
