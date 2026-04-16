# FR02 机器人强化学习训练文档

> 本文档记录了使用 LeggedLab + Isaac Lab 框架对 FR02 微型人形机器人进行强化学习训练的完整流程、参数与经验总结。

---

## 一、环境信息

| 项目 | 版本 / 路径 |
|------|------------|
| 操作系统 | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 5070 (12 GB VRAM) |
| CUDA Driver | 580.126.09 |
| Isaac Sim | 5.1 |
| Isaac Lab | `/home/rob/isaac_workspace/IsaacLab` |
| LeggedLab | `/home/rob/isaac_workspace/LeggedLab` |
| Conda 环境 | `isaac`（`~/miniconda3/envs/isaac`）|
| Python | 3.11 |
| rsl-rl-lib | >= 4.0.0 |

---

## 二、机器人基本参数

| 参数 | 值 |
|------|-----|
| 机器人名称 | FR02 (`fr02_urdf_02_0404`) |
| 总质量 | ~1.2 kg |
| 高度 | ~50 cm |
| 关节数量 | 27 个旋转关节 |
| URDF 路径 | `/home/rob/isaac_workspace/urdf/0408/fr.urdf` |
| USD 路径 | `legged_lab/assets/fr/fr02/fr02.usd` |

### 关节结构

| 部位 | 关节数 | 关节名称 |
|------|--------|---------|
| 左腿 | 6 | l_hip_pitch/roll/yaw, l_knee_pitch, l_ankle_pitch/roll |
| 右腿 | 6 | r_hip_pitch/roll/yaw, r_knee_pitch, r_ankle_pitch/roll |
| 腰胸 | 3 | waist_yaw, chest_roll, chest_pitch |
| 左臂 | 5 | l_shoulder_pitch/roll, l_upper_arm_yaw, l_elbow_pitch, l_wrist_roll |
| 右臂 | 5 | r_shoulder_pitch/roll, r_upper_arm_yaw, r_elbow_pitch, r_wrist_roll |
| 头部 | 2 | head_yaw, head_pitch |

### 执行器参数（实际机器人规格）

| 关节 | 刚度 (N·m/rad) | 阻尼 (N·m·s/rad) |
|------|---------------|-----------------|
| 腿部关节（hip/knee/ankle） | 9.156 | 0.305 |
| 肩部关节（shoulder_pitch/roll） | 1.831 | 0.092 |
| 上臂/肘部（upper_arm_yaw/elbow_pitch） | 1.831 | 0.092 |
| 手腕（wrist_roll） | 1.831 | 0.031 |
| 胸部（chest_pitch/roll） | 2.442 | 0.153 |
| 腰部（waist_yaw） | 4.883 | 0.153 |
| 头部（head_yaw/pitch） | 1.831 | 0.092 |

---

## 三、任务说明

### 任务名称：`fr02_plane`

在**纯平面地形（无起伏）** 上训练 FR02 双足机器人行走，作为基准对比。

| 任务 | 地形 | 用途 |
|------|------|------|
| `fr02_plane` | 纯平面（plane） | 基准训练，快速验证策略可行性 |
| `fr02_flat` | 砾石地形（gravel） | 轻度不规则地面 |
| `fr02_rough` | 粗糙地形（rough） | 强泛化能力训练，启用高度扫描 |

---

## 四、URDF 预处理

训练前对原始 URDF 做了以下修改：

1. **移除 MuJoCo 标签**：`<mujoco>` 块与 Isaac Sim 不兼容，已删除
2. **清理 mass 尾部空格**：SolidWorks 导出的瑕疵（`base_link`、`l_knee_pitch_link`、`head_pitch_link`）
3. **URDF 转 USD**：使用 `IsaacLab/scripts/tools/convert_urdf.py` 转换

```bash
/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p \
  /home/rob/isaac_workspace/IsaacLab/scripts/tools/convert_urdf.py \
  /home/rob/isaac_workspace/urdf/0408/fr.urdf \
  /home/rob/isaac_workspace/LeggedLab/legged_lab/assets/fr/fr02/fr02.usd \
  --headless
```

---

## 五、与 G1 的关键差异

FR02 接入 LeggedLab 时需要注意与 G1 的关节命名差异：

| 功能 | G1 | FR02 |
|------|-----|------|
| 膝关节 | `.*_knee_joint` | `.*_knee_pitch_joint` |
| 腰部 | `waist_yaw/roll/pitch_joint` | `waist_yaw_joint` + `chest_roll/pitch_joint` |
| 上臂旋转 | `.*_shoulder_yaw_joint` | `.*_upper_arm_yaw_joint` |
| 肘关节 | `.*_elbow_joint` | `.*_elbow_pitch_joint` |
| 手腕 | 3 DOF (yaw/roll/pitch) | 1 DOF (roll only) |
| 左右前缀 | `left_`/`right_` | `l_`/`r_` |
| 躯干（body_names） | `.*torso.*` | `.*chest_pitch.*` |

---

## 六、训练命令

### 激活环境

```bash
source ~/miniconda3/bin/activate isaac
cd /home/rob/isaac_workspace/LeggedLab
export TERM=xterm
```

### 启动训练（后台运行）

```bash
nohup /home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/train.py \
  --task=fr02_plane \
  --headless \
  --logger=tensorboard \
  --num_envs=2048 \
  --max_iterations=25000 \
  >> logs/train_fr02_plane_25k.log 2>&1 &

echo "训练 PID: $!"
```

### 实时查看训练日志

```bash
tail -f logs/train_fr02_plane_25k.log
```

### 查看 TensorBoard

```bash
tensorboard --logdir logs/fr02_plane
# 浏览器打开 http://localhost:6006
```

---

## 七、推理（Play）命令

### Headless 模式

```bash
/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/play.py \
  --task=fr02_plane \
  --headless \
  --num_envs=16 \
  --load_run=2026-04-16_15-14-34 \
  --checkpoint=model_24999.pt
```

### GUI 可视化模式

```bash
pkill -f 'play.py\|train.py'
sleep 3

export TERM=xterm DISPLAY=:1

/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/play.py \
  --task=fr02_plane \
  --num_envs=16 \
  --load_run=2026-04-16_15-14-34 \
  --checkpoint=model_24999.pt
```

---

## 八、训练结果（fr02_plane 25000 iter）

### 基本信息

| 项目 | 数值 |
|------|------|
| 训练日期 | 2026-04-16 |
| 迭代次数 | 25000 |
| 环境数量 | 2048 |
| 单步迭代时间 | ~0.67～0.78 s |
| 总训练时间 | **4 小时 53 分钟** |
| 大约吞吐量 | ~63,000 steps/s |

### 最终性能指标

| 指标 | 最终值 | 说明 |
|------|--------|------|
| **Mean Reward** | **7.01** | 综合奖励 |
| **Mean Episode Length** | **893** | 接近满帧 1000 |
| Mean Value Loss | ~0.10 | Critic 损失 |
| Mean Action Std | ~0.45 | 动作标准差 |

### 分项奖励

| 奖励项 | 最终值 | 含义 |
|--------|--------|------|
| `track_lin_vel_xy_exp` | **0.7822** | 线速度跟踪（78.2%） |
| `track_ang_vel_z_exp` | **0.5241** | 角速度跟踪（52.4%） |
| `termination_penalty` | **-0.0333** | 极少跌倒 |
| `feet_air_time` | 0.0076 | 迈步步态奖励 |
| `energy` | -0.0046 | 能量消耗 |
| `action_rate_l2` | -0.5070 | 动作平滑性惩罚 |
| `flat_orientation_l2` | -0.0176 | 躯干保持水平 |
| `body_orientation_l2` | -0.0387 | 身体姿态惩罚 |
| `joint_deviation_arms` | -0.1639 | 手臂偏离默认姿态 |
| `joint_deviation_hip` | -0.1184 | 髋关节偏离 |
| `joint_deviation_legs` | -0.0201 | 腿部关节偏离 |
| `undesired_contacts` | -0.0673 | 非足部接触地面 |
| `feet_slide` | -0.0185 | 足部滑动 |
| `feet_force` | -0.0007 | 足部冲击力 |
| `dof_pos_limits` | -0.1547 | 关节超限惩罚 |

### 训练收敛过程

| 阶段 | 迭代 | Mean Reward | Episode Length | 线速度跟踪 |
|------|------|-------------|---------------|-----------|
| 初始 | 1 | -10.55 | 88 | 3.4% |
| 20 分钟 | 2008 | -7.18 | 663 | 47.4% |
| 40 分钟 | 3613 | -1.55 | 867 | 69.4% |
| 1 小时 | 5238 | +0.82 | 841 | 74.3% |
| 2 小时 | 11623 | +4.99 | 893 | 79.0% |
| 3 小时 | 16908 | +6.25 | 901 | 75.6% |
| 4 小时 | 23318 | +4.53 | 948 | 81.3% |
| 最终 | 25000 | +7.01 | 893 | 78.2% |

### 导出文件

```
logs/fr02_plane/2026-04-16_15-14-34/
├── model_24999.pt              # 最终权重（PyTorch）
├── exported/
│   ├── policy.pt               # TorchScript JIT（部署用）
│   └── policy.onnx             # ONNX 格式（跨框架部署）
└── events.out.tfevents.*       # TensorBoard 日志
```

---

## 九、仿真配置

| 参数 | 值 |
|------|-----|
| `dt` | 0.005 s（200 Hz 物理步） |
| `decimation` | 4（控制频率 50 Hz） |
| `max_episode_length_s` | 20 s |
| `env_spacing` | 2.5 m |
| `init_state.pos.z` | 0.35 m |
| `action_scale` | 0.25 |
| domain_rand mass | (-0.3, 0.3) kg |
| `terminate_contacts_body_names` | `.*chest_pitch.*` |
| `feet_body_names` | `.*ankle_roll.*` |

---

## 十、已知问题与后续改进

1. **肩膀姿态问题**：初始状态下手臂水平伸展，应改为自然下垂（需调整 `shoulder_roll` 默认角度）
2. **执行器参数优化**：当前使用估算值，应替换为实际机器人刚度/阻尼参数
3. **增加迭代次数**：50000 次迭代可进一步提升性能
4. **粗糙地形训练**：在 plane 基础上进行 `fr02_rough` 训练，增强泛化能力

---

## 十一、文件结构

```
legged_lab/
├── assets/
│   └── fr/
│       ├── __init__.py          # 导出 FR02_CFG
│       ├── fr.py                # ArticulationCfg 定义
│       └── fr02/
│           ├── fr02.usd         # URDF 转换生成
│           ├── config.yaml      # 转换元数据
│           └── configuration/   # USD 子文件
└── envs/
    └── fr02/
        ├── __init__.py
        └── fr02_config.py       # Reward + EnvCfg + AgentCfg
```

---

*文档生成日期：2026-04-16 | 训练 run ID：`2026-04-16_15-14-34`*
