# VLA 微调：π0 / SmolVLA 在本项目上怎么落地

> 结论先说：**π0 不适合在你的机器上微调，得上云 GPU。** 本机能试的是 SmolVLA（450M）。
> 数据这一层（图像 + 语言 + LeRobotDataset）已经做好了，两条路都能直接用。

## 1. 硬件对照（你的机器：RTX 4060 Laptop，8GB 显存）

| 方案 | 参数量 | 微调显存 | 你这台能不能干 |
|---|---|---|---|
| **π0** 全量微调 | 3.3B | 官方 openpi 配方 8×A100(80G) | ✗ |
| **π0** LoRA 微调 | 3.3B | ≥24GB（社区实践多在 A100 40G 上） | ✗ |
| **π0** 纯推理 | 3.3B | 8~10GB（bf16） | 勉强，容易 OOM |
| **SmolVLA** 微调 | 450M | 官方给单卡配方；LoRA/冻结视觉塔后 8~16GB 可能塞下 | 可以试，很紧 |
| **ACT / Diffusion Policy** | 10~100M | 4~8GB | ✓ |
| 本仓库的 MLP（现状基线） | 0.5M | 不需要 GPU | ✓ 已跑通 |

要点：
- π0 是 3.3B 的 flow-matching VLA，**权重本身就 ~7GB（bf16）**，微调还要优化器状态和激活 —— 8GB 显存差得远。
- 租云 GPU 是标准做法：A100 40G 大约 ¥5~10/小时。我们这种"几十段轨迹"的小数据集，LoRA 微调 1~2 小时就能出结果，成本几十块。
- 想在本地玩 VLA，**SmolVLA 是唯一现实选项**（LeRobot 官方就是为了单卡/消费级卡做的）。

## 2. 数据：已经能直接喂 VLA

```bash
# 用脚本专家生成（不需要人遥控），50 段大约 1~2 分钟
python collect_data_vla.py --episodes 50 --out_root data/grasp_vla
python collect_data_vla.py --episodes 200 --image_size 224 --overwrite   # 想更多/更小分辨率
```

产出是标准 **LeRobotDataset**（已验证可读回）：

| key | 形状 | 说明 |
|---|---|---|
| `observation.images.top` | (256,256,3) | 场景相机，看得到方块与蓝盘 |
| `observation.images.wrist` | (256,256,3) | 腕部相机，贴近夹爪与物块 |
| `observation.state` | (7,) | 关节1..6 + 夹爪角（**不含**物块位姿，逼策略看图） |
| `action` | (7,) | 目标关节1..6 + 夹爪命令 |
| `task` | str | `put the yellow cube on the blue disc` |

fps=20（对应 50ms 控制周期）。**动作语义**：`action` = 未来 0.2s 的实测状态（与仓库里 MLP 数据同一套约定，见 `src/expert_grasp.py` 文件头）—— 对预测动作块的 VLA 来说正好是一条未来轨迹。

想验证相机拍到了什么：`vla_cams_check.png`（脚本导出的几帧）。

## 3. 环境

现在两个环境的状况（都不满足 VLA 训练）：

| 环境 | torch | lerobot | transformers | 问题 |
|---|---|---|---|---|
| `lerobot_mujoco_vla_tutorial` | 2.7.1 **CPU** | 0.4.3 | ✗ 缺 | `huggingface_hub` 版本过旧，`lerobot.policies.factory` 直接 import 失败 |
| `lerobot` | 2.6.0+cu124 ✓ | 0.1.0 | 4.50.3 | lerobot 太老，没有 π0/SmolVLA |

新建一个干净环境（**本机 SmolVLA 用**）：

```bash
conda create -n vla python=3.10 -y
conda activate vla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "lerobot[smolvla]" "transformers>=4.57.1" "huggingface_hub>=0.34.2,<0.36.0"
# Windows 上不用装 flash-attn（可选依赖，会尝试编译）
```

π0 额外要求：lerobot 0.4.3 的 `pi` extra 指向**一个 git 分支的 transformers**：

```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```

（这也是为什么 π0 这条路更适合在云上按 openpi 官方仓库走。）

## 4. 微调

### 4.1 本地试 SmolVLA

```bash
lerobot-train \
  --dataset.repo_id=uranus_grasp_vla \
  --dataset.root=data/grasp_vla \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --output_dir=outputs/train/uranus_smolvla \
  --job_name=uranus_smolvla \
  --policy.device=cuda --wandb.enable=false \
  --batch_size=2 --steps=20000
```

8GB 显存下的调参顺序：`batch_size=1` → `--policy.freeze_vision_encoder=true` → 图像降到 224/192 → `--policy.train_expert_only=true`（只训动作专家，最省）。`--help` 确认你那个版本的具体字段名。

### 4.2 云端微调 π0（推荐路线）

1. 租一台 A100 40G（AutoDL / 恒源云 / RunPod 都行），装 CUDA 版 torch + lerobot；
2. 把 `data/grasp_vla/` 整个目录传上去（几十 MB~几百 MB）；
3. 两条可选路线：
   - **LeRobot**：`--policy.type=pi0 --policy.pretrained_path=lerobot/pi0_base`，其余同上（记得先装那个 git 分支的 transformers）；
   - **openpi（官方）**：按 openpi 仓库的 LoRA 配方，把它指向转换好的 LeRobotDataset；
4. 训完把 `pretrained_model/` 目录拉回本地评测。

## 5. 评测

```bash
# 本机自检（不需要 lerobot 策略，验证评测回路）
python eval_vla.py --dummy_mlp ckpt/manual/policy.pt --num_episodes 3

# 评测真策略（在装了 lerobot 策略依赖的环境里）
python eval_vla.py --ckpt outputs/train/uranus_smolvla/checkpoints/last/pretrained_model \
                   --num_episodes 50 --images top,wrist
```

评测脚本给策略的输入和训练时完全一致（两路图像 + 7 维本体状态 + 语言），执行方式也是 20Hz、动作块前 8 步；判据与其它脚本统一（抬升 > 2cm 且 `env.success()`）。

## 6. 现状与预期（说实话）

- **数据管线**：✅ 已就绪并验证。已生成 **`data/grasp_vla/` = 300 段 / 50700 帧 / 1.1GB**
  （224² 两路相机，fps=20，专家成功率 100%，生成耗时 27 分钟），读回校验通过
  （`observation.images.top/wrist` (3,224,224)、`observation.state` (7,)、`action` (7,)、`task` 字符串）。
  > 注意：`LeRobotDataset.create()` 要求目标目录不存在，所以**采集不能断点续采** ——
  > 中途被中断就得删掉重来（这也是为什么一次跑 300 段要留足 30 分钟）。
- **评测管线**：✅ 回路已用 MLP 自检跑通；真策略那条路要等 VLA 环境建好
  （接口集中在 `eval_vla.py::load_policy`，换 lerobot 版本只改那一处）。
- **本机已有基线（都在当前 4cm 场景上重训过）**：
  | 数据 | 策略 | 评测成功率 |
  |---|---|---|
  | 脚本专家 200 段（轻量 npz） | MLP 行为克隆 | **1/20 = 5%** |
  | 人手遥操作 6 段（轻量 npz） | MLP 行为克隆 | **0/20 = 0%**（lift 1.2mm，根本没抓到） |
  | 脚本专家 300 段（VLA 图像格式） | 未训练 | — |
  **所以别指望"VLA 一上就好了"** —— 数据量、任务难度、评价精度都还卡在那里。
- **要出效果，性价比最高的顺序**：
  1. 数据已经攒到 300 段了（VLA 格式）✓；想再加就再跑一次 `collect_data_vla.py`（换 `--out_root` 或 `--overwrite`）；
  2. 本地试 SmolVLA（十几分钟能看出 loss 有没有在降）；
  3. 还不行再上云摸 π0（π0 的先验更强，但成本和数据要求也更高）；
  4. 同时把任务放宽（更大的盘子/浅盒、物块离盘子更近），让成功率这件事先有正反馈。
