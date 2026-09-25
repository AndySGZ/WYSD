# debug/ —— 2026-09-11 那轮抓取联调的调试脚本（保留存档，不是主流程）

这些脚本是当时定位"夹不住 / 抬不起来 / 放不准"时一个个试出来的，**结论都已经写进
`src/env_grasp.py` 的注释和 README 里**了。留着是为了以后换物块/换夹爪时能照着再查一遍。

它们都用绝对路径 `sys.path.insert(0, .../Lerobot-Uranus-VLA-Demo)`，所以放在子目录里也能直接跑：

```bash
python debug/grasp_ik6.py        # 例：6 自由度齿面中点 IK 的对齐检查
```

| 脚本 | 当时在查什么 | 结论（已落地到代码） |
|---|---|---|
| `_tmp_debug*.py` / `_tmp_verify*.py` | 模型能不能加载、关节限位/索引对不对 | 模型与场景 keyframe 对齐 |
| `_tmp_gravcomp*.py` | 重力垂沉多少、要不要做重力补偿 | 垂沉 ~1cm → `env_grasp.move_to()` 闭环修正 |
| `_tmp_sweep.py` / `_tmp_opt.py` / `_tmp_solver.py` | 求解器/接触参数（impratio、solref、积分器） | 齿面+物块 `solref=0.002`（软接触会把齿压进物块 2cm） |
| `_tmp_gripper.py` / `_tmp_grip_poses.py` / `_tmp_jawlimit.py` / `_tmp_jawstiff.py` | 夹爪行程/刚度/齿面间隙标定 | 齿面净距 ≈ 12.9mm + 246mm/rad；`grasp_angle()` 的标定来源 |
| `_tmp_tcp_vs_tooth.py` | TCP 与齿面中点的偏差 | 抓取对齐必须用"齿面中点"而不是 TCP |
| `_tmp_noobj.py` | 没物块时的空抓行为 | 确认夹爪几何 |
| `grasp_probe*.py` / `grasp_geom.py` / `grasp_orient*.py` / `grasp_tilt.py` | 齿面朝向、张开时齿面不平行 | → `align_jaws()` 平行补偿 |
| `grasp_descend.py` / `grasp_sag.py` / `grasp_contact.py` / `grasp_debug.py` | 下探到哪、什么时候接触、为什么顶住 | → `GRASP_PITCH=10°`（否则夹爪本体顶桌面） |
| `grasp_ik5.py` / `grasp_ik6.py` | 5 vs 6 自由度 IK | 6 自由度（位置+姿态），见 `solve_ik_grasp()` |
| `grasp_jawtest*.py` / `grasp_jawkp.py` / `grasp_kpsweep.py` / `grasp_gripcheck.py` / `grasp_close2.py` | 闭爪力、kp、有没有真的夹住 | → 夹持角按物块宽度反解 + 接触调硬 |
| `grasp_gravcomp*.py` | 抓着重物时要不要补偿 | 不需要（0.1kg 物块） |
| `grasp_e2e2.py` | 早期端到端（已被顶层 `grasp_e2e.py` 取代） | 主流程用 `grasp_e2e.py --randomize N` |

> 主流程的端到端验证是顶层的 `grasp_e2e.py`（专家 + 随机化回归），不要用这里的 `grasp_e2e2.py`。
