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

### 各关节 URDF 参数（轴向、限位、力矩、速度）

| 关节名 | 轴向 | 下限 (rad) | 上限 (rad) | 力矩 (Nm) | 速度 (rad/s) |
|--------|------|-----------|-----------|----------|-------------|
| l_hip_pitch_joint | 0 -1 0 | -2.40 | 2.40 | 1.6 | 6.283 |
| l_hip_roll_joint | 1 0 0 | -0.20 | 1.46 | 1.6 | 6.283 |
| l_hip_yaw_joint | 0 0 -1 | -1.57 | 1.57 | 0.4 | 10.472 |
| l_knee_pitch_joint | 0 1 0 | 0.00 | 1.95 | 1.6 | 6.283 |
| l_ankle_pitch_joint | 0 1 0 | -0.37 | 0.52 | 1.6 | 6.283 |
| l_ankle_roll_joint | 1 0 0 | -0.52 | 0.47 | 1.2 | 6.283 |
| r_hip_pitch_joint | 0 1 0 | -2.40 | 2.40 | 1.6 | 6.283 |
| r_hip_roll_joint | 1 0 0 | -1.46 | 0.20 | 1.6 | 6.283 |
| r_hip_yaw_joint | 0 0 -1 | -1.57 | 1.57 | 0.4 | 10.472 |
| r_knee_pitch_joint | 0 1 0 | 0.00 | 1.95 | 1.6 | 6.283 |
| r_ankle_pitch_joint | 0 1 0 | -0.37 | 0.52 | 1.6 | 6.283 |
| r_ankle_roll_joint | 1 0 0 | -0.47 | 0.52 | 1.2 | 6.283 |
| waist_yaw_joint | 0 0 -1 | -0.60 | 0.60 | 1.2 | 6.283 |
| chest_roll_joint | 1 0 0 | -0.27 | 0.27 | 1.2 | 6.283 |
| chest_pitch_joint | 0 1 0 | 0.00 | 0.82 | 1.2 | 6.283 |
| l_shoulder_pitch_joint | 0 -1 0 | -0.80 | 3.00 | 0.4 | 5.236 |
| l_shoulder_roll_joint | 1 0 0 | -1.47 | 0.62 | 0.4 | 5.236 |
| l_upper_arm_yaw_joint | 0 -1 0 | -1.40 | 1.40 | 0.2 | 8.378 |
| l_elbow_pitch_joint | 0 0 -1 | 0.00 | 2.20 | 0.4 | 5.236 |
| l_wrist_roll_joint | -1 0 0 | -2.10 | 2.10 | 0.14 | 4.189 |
| r_shoulder_pitch_joint | 0 1 0 | -3.00 | 0.80 | 0.4 | 5.236 |
| r_shoulder_roll_joint | 1 0 0 | -0.62 | 1.47 | 0.4 | 5.236 |
| r_upper_arm_yaw_joint | 0 1 0 | -1.40 | 1.40 | 0.2 | 8.378 |
| r_elbow_pitch_joint | 0 0 -1 | -2.20 | 0.00 | 0.4 | 5.236 |
| r_wrist_roll_joint | 1 0 0 | -2.10 | 2.10 | 0.14 | 4.189 |
| head_yaw_joint | 0 0 -1 | -1.00 | 1.00 | 0.2 | 8.378 |
| head_pitch_joint | 0 1 0 | -0.87 | 0.52 | 0.4 | 6.283 |

### 初始关节角度（init_state）

| 关节 | 初始角度 (rad) | 说明 |
|------|---------------|------|
| `.*_hip_pitch_joint` | -0.20 | 髋关节微屈 |
| `.*_knee_pitch_joint` | 0.42 | 膝关节弯曲 |
| `.*_ankle_pitch_joint` | -0.23 | 踝关节配合膝弯 |
| `l_shoulder_roll_joint` | -1.30 | 左臂自然下垂（~75度） |
| `r_shoulder_roll_joint` | +1.30 | 右臂自然下垂（~75度） |
| 其余关节 | 0.0 | 默认零位 |
| 初始高度 pos.z | 0.35 m | 腿长 ~0.30m + 余量 |

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

## 九、仿真与 MDP 配置

### 仿真参数

| 参数 | 值 |
|------|-----|
| `dt` | 0.005 s（200 Hz 物理步） |
| `decimation` | 4（控制频率 50 Hz） |
| `max_episode_length_s` | 20 s（训练）/ 40 s（推理） |
| `env_spacing` | 2.5 m |
| `init_state.pos.z` | 0.35 m |
| `soft_joint_pos_limit_factor` | 0.90 |
| `gpu_max_rigid_patch_count` | 327680 |

### 观测与动作

| 参数 | 值 |
|------|-----|
| `action_scale` | 0.25 |
| `actor_obs_history_length` | 10 帧（plane/flat）/ 1 帧（rough） |
| `critic_obs_history_length` | 10 帧（plane/flat）/ 1 帧（rough） |
| `clip_observations` | 100.0 |
| `clip_actions` | 100.0 |

### 速度指令范围（Commands）

| 参数 | 范围 |
|------|------|
| `lin_vel_x` | (-0.6, 1.0) m/s |
| `lin_vel_y` | (-0.5, 0.5) m/s |
| `ang_vel_z` | (-1.57, 1.57) rad/s |
| `heading` | (-pi, pi) |
| `resampling_time_range` | (10.0, 10.0) s |
| `rel_standing_envs` | 0.2（20% 环境给静止指令） |
| `heading_control_stiffness` | 0.5 |

### 观测噪声

| 参数 | 噪声尺度 |
|------|---------|
| `ang_vel` | 0.2 |
| `projected_gravity` | 0.05 |
| `joint_pos` | 0.01 |
| `joint_vel` | 1.5 |
| `height_scan` | 0.1 |

### 域随机化（Domain Randomization）

| 参数 | 值 |
|------|-----|
| 摩擦系数（静摩擦） | (0.6, 1.0) |
| 摩擦系数（动摩擦） | (0.4, 0.8) |
| 恢复系数 | (0.0, 0.005) |
| 基座质量扰动 | (-0.3, 0.3) kg（作用于 `chest_pitch_link`） |
| 推力扰动间隔 | (10.0, 15.0) s |
| 推力速度范围 | x: (-1.0, 1.0), y: (-1.0, 1.0) m/s |
| 关节重置位置范围 | (0.5, 1.5) 倍默认值 |

### 奖励函数配置（FR02RewardCfg）

| 奖励项 | 函数 | 权重 | 关键参数 |
|--------|------|------|---------|
| `track_lin_vel_xy_exp` | 线速度跟踪（yaw frame） | +1.0 | std=0.5 |
| `track_ang_vel_z_exp` | 角速度跟踪（world frame） | +1.0 | std=0.5 |
| `lin_vel_z_l2` | Z 轴线速度惩罚 | -1.0 | - |
| `ang_vel_xy_l2` | XY 角速度惩罚 | -0.05 | - |
| `energy` | 能量消耗 | -1e-3 | - |
| `dof_acc_l2` | 关节加速度惩罚 | -2.5e-7 | - |
| `action_rate_l2` | 动作变化率惩罚 | -0.01 | - |
| `undesired_contacts` | 非足部触地 | -1.0 | body: `(?!.*ankle.*).*`, threshold=1.0 |
| `fly` | 双脚离地惩罚 | -1.0 | body: `.*ankle_roll.*`, threshold=1.0 |
| `body_orientation_l2` | 躯干姿态偏差 | -2.0 | body: `.*chest_pitch.*` |
| `flat_orientation_l2` | 水平姿态偏差 | -1.0 | - |
| `termination_penalty` | 跌倒终止重罚 | -200.0 | - |
| `feet_air_time` | 双足交替步态 | +0.15 | body: `.*ankle_roll.*`, threshold=0.4 |
| `feet_slide` | 足部滑动 | -0.25 | body: `.*ankle_roll.*` |
| `feet_force` | 足部冲击力 | -3e-3 | threshold=50, max_reward=40 |
| `feet_too_near` | 双脚过近 | -2.0 | body: `.*ankle_roll.*`, threshold=0.06 |
| `feet_stumble` | 足部绊倒 | -2.0 | body: `.*ankle_roll.*` |
| `dof_pos_limits` | 关节超限 | -2.0 | - |
| `joint_deviation_hip` | 髋/肩/肘偏差 | -0.15 | joints: `hip_yaw/roll, shoulder_pitch, elbow_pitch` |
| `joint_deviation_arms` | 腰/肩/腕偏差 | -0.2 | joints: `waist, chest, shoulder_roll, upper_arm_yaw, wrist` |
| `joint_deviation_legs` | 腿部关节偏差 | -0.02 | joints: `hip_pitch, knee_pitch, ankle` |

### 环境覆盖参数（FR02 相对 BaseEnvCfg 的定制）

| 参数 | FR02 值 | 基类默认值 |
|------|---------|----------|
| `height_scanner.prim_body_name` | `chest_pitch_link` | MISSING |
| `terminate_contacts_body_names` | `.*chest_pitch.*` | MISSING |
| `feet_body_names` | `.*ankle_roll.*` | MISSING |
| `domain_rand.add_base_mass.body_names` | `.*chest_pitch.*` | MISSING |
| `domain_rand.add_base_mass.mass_distribution_params` | (-0.3, 0.3) | (-5.0, 5.0) |

### PPO 算法超参数

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 24 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 1e-3 |
| `schedule` | adaptive |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.005 |
| `desired_kl` | 0.01 |
| `max_grad_norm` | 1.0 |
| `save_interval` | 100 |

### 策略网络架构

```
Actor:  MLP [obs_dim → 512 → 256 → 128 → 27]  (ELU 激活)
Critic: MLP [obs_dim → 512 → 256 → 128 → 1]   (ELU 激活)
观测历史长度: actor=10, critic=10 帧 (plane/flat)
```

---

## 十、训练结果 v2（fr02_plane 50000 iter，肩膀下垂 + 实际刚度参数）

### 改进内容

- 肩膀初始姿态：`l_shoulder_roll` 从 0.18 改为 **-1.3**，`r_shoulder_roll` 从 -0.18 改为 **+1.3**（手臂自然下垂）
- 执行器参数：从估算值替换为实际机器人刚度/阻尼规格

### 基本信息

| 项目 | 数值 |
|------|------|
| 训练日期 | 2026-04-16 ~ 2026-04-17 |
| 迭代次数 | 50000 |
| 环境数量 | 2048 |
| 单步迭代时间 | ~0.68～0.96 s |
| 总训练时间 | **10 小时 7 分钟** |
| Run ID | `2026-04-16_21-36-43` |

### 最终性能指标

| 指标 | 最终值 |
|------|--------|
| **Mean Reward** | **-0.38** |
| **Mean Episode Length** | **881** |
| 线速度跟踪 | **74.9%** |
| 角速度跟踪 | **29.5%** |
| 跌倒惩罚 | -0.052 |

### 训练收敛过程

| 阶段 | 迭代 | Mean Reward | Episode Length | 线速度跟踪 |
|------|------|-------------|---------------|-----------|
| 30 分钟 | 2508 | -12.83 | 704 | 32.7% |
| 1 小时 | 4843 | -8.81 | 703 | 58.8% |
| 1.5 小时 | 7428 | -7.06 | 761 | 66.9% |
| 2 小时 | 9843 | -5.83 | 795 | 66.8% |
| 2.5 小时 | 13273 | -4.36 | 843 | 75.9% |
| 3.5 小时 | 20298 | -3.93 | 798 | 70.6% |
| 5 小时 | 35233 | -3.04 | 820 | 69.1% |
| 5.5 小时 | 37603 | -1.96 | 820 | 78.0% |
| 6.5 小时 | 44673 | -1.07 | 851 | 74.2% |
| 7 小时 | 47048 | -0.59 | 844 | 75.4% |
| 最终 | 50000 | -0.38 | 881 | 74.9% |

### v1 vs v2 对比

| 指标 | v1 (25k, 估算参数) | v2 (50k, 实际参数) |
|------|-------------------|-------------------|
| Mean Reward | +7.01 | -0.38 |
| Episode Length | 893 | 881 |
| 线速度跟踪 | 78.2% | 74.9% |
| 角速度跟踪 | 52.4% | 29.5% |
| 肩膀姿态 | 水平伸展 | 自然下垂 |
| 执行器参数 | 估算值 | 实际规格 |

> v2 奖励数值较低是因为实际刚度参数更真实，肩膀下垂后 joint_deviation 基线变化。Play 效果 OK。

### 导出文件

```
logs/fr02_plane/2026-04-16_21-36-43/
├── model_49999.pt              # 最终权重（PyTorch）
└── events.out.tfevents.*       # TensorBoard 日志
```

### 推理命令

```bash
/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/play.py \
  --task=fr02_plane --num_envs=16 \
  --load_run=2026-04-16_21-36-43 --checkpoint=model_49999.pt
```

---

## 十一、已知问题与后续改进

1. ~~**肩膀姿态问题**~~：已修复，`shoulder_roll` 改为 ±1.3 rad（自然下垂）
2. ~~**执行器参数优化**~~：已替换为实际机器人刚度/阻尼参数
3. **角速度跟踪偏低**：v2 仅 29.5%，可尝试增大 `track_ang_vel_z_exp` 权重至 1.5
4. **粗糙地形训练**：在 plane 基础上进行 `fr02_rough` 训练，增强泛化能力
5. **action_rate_l2 惩罚较大**：可适当降低权重或增大 `action_scale`
6. **大文件**：`fr02_base.usd`（60MB）建议后续迁移至 Git LFS

---

## 十二、文件结构

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

*文档更新日期：2026-04-17 | 训练 run ID：v1 `2026-04-16_15-14-34` / v2 `2026-04-16_21-36-43`*
