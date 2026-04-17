# FR02 Sim-to-Real 部署路线图

> 本文档记录从仿真训练到 FR02 实机部署的完整路径，包括已完成环节、缺失环节、潜在的坑和逐步实施建议。

---

## 一、当前状态

- 仿真训练完成：50k 迭代，plane 地形，实际刚度/阻尼参数，肩膀自然下垂
- 策略已导出：`policy.pt`（TorchScript JIT）和 `policy.onnx`
- Play 验证效果 OK

---

## 二、策略导出验证结果

### 文件位置

```
logs/fr02_plane/2026-04-16_21-36-43/exported/
├── policy.pt    (2.5 MB, TorchScript JIT)
└── policy.onnx  (2.5 MB, ONNX opset 18)
```

### 维度确认

| 项目 | 值 |
|------|-----|
| 输入维度 | **900** = 90（单帧）x 10（历史帧） |
| 输出维度 | **27**（各关节动作值） |
| ONNX 输入名 | `obs` |
| ONNX 输出名 | `actions` |
| 归一化 | 模型**不包含**obs归一化，需在外部实现 |

### 单帧观测向量构造（90 维）

```
obs = concat([
    ang_vel * 1.0,              # [0:3]   IMU 角速度 (body frame, rad/s)
    projected_gravity * 1.0,    # [3:6]   重力投影 (body frame)
    command * 1.0,              # [6:9]   速度指令 (vx, vy, wz)
    joint_pos * 1.0,            # [9:36]  关节位置 - 默认位置 (rad)
    joint_vel * 1.0,            # [36:63] 关节速度 (rad/s)
    last_action * 1.0,          # [63:90] 上一步动作输出
])
```

### 动作后处理

```python
target_joint_pos = clip(action, -100, 100) * 0.25 + default_joint_pos
```

---

## 三、仿真中的关节顺序（27 维，索引 0-26）

**这是 sim-to-real 最关键的映射表，真机侧必须严格按此顺序构造 joint_pos/joint_vel 和解析 action。**

| 索引 | 关节名 | 默认角度 (rad) |
|------|--------|---------------|
| 0 | l_hip_pitch_joint | -0.20 |
| 1 | r_hip_pitch_joint | -0.20 |
| 2 | waist_yaw_joint | 0.0 |
| 3 | l_hip_roll_joint | 0.0 |
| 4 | r_hip_roll_joint | 0.0 |
| 5 | chest_roll_joint | 0.0 |
| 6 | l_hip_yaw_joint | 0.0 |
| 7 | r_hip_yaw_joint | 0.0 |
| 8 | chest_pitch_joint | 0.0 |
| 9 | l_knee_pitch_joint | 0.42 |
| 10 | r_knee_pitch_joint | 0.42 |
| 11 | head_yaw_joint | 0.0 |
| 12 | l_shoulder_pitch_joint | 0.0 |
| 13 | r_shoulder_pitch_joint | 0.0 |
| 14 | l_ankle_pitch_joint | -0.23 |
| 15 | r_ankle_pitch_joint | -0.23 |
| 16 | head_pitch_joint | 0.0 |
| 17 | l_shoulder_roll_joint | -1.30 |
| 18 | r_shoulder_roll_joint | 1.30 |
| 19 | l_ankle_roll_joint | 0.0 |
| 20 | r_ankle_roll_joint | 0.0 |
| 21 | l_upper_arm_yaw_joint | 0.0 |
| 22 | r_upper_arm_yaw_joint | 0.0 |
| 23 | l_elbow_pitch_joint | 0.0 |
| 24 | r_elbow_pitch_joint | 0.0 |
| 25 | l_wrist_roll_joint | 0.0 |
| 26 | r_wrist_roll_joint | 0.0 |

---

## 四、真机控制循环伪代码

```python
import torch
import numpy as np
from collections import deque

# 初始化
policy = torch.jit.load("policy.pt")
policy.eval()

HISTORY_LEN = 10
OBS_DIM = 90
ACTION_DIM = 27
ACTION_SCALE = 0.25
DT = 0.02  # 50 Hz

# 默认关节角度 (27维, 按上表索引顺序)
default_joint_pos = np.array([
    -0.20, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.42, 0.42, 0.0, 0.0, 0.0, -0.23, -0.23, 0.0,
    -1.30, 1.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])

obs_history = deque([np.zeros(OBS_DIM)] * HISTORY_LEN, maxlen=HISTORY_LEN)
last_action = np.zeros(ACTION_DIM)

def quat_rotate_inverse(q, v):
    """从世界坐标系的重力向量投影到body坐标系"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # ... 四元数旋转逆 ...

while running:
    t0 = time.time()

    # 1. 读传感器
    ang_vel = imu.get_angular_velocity()        # [3] body frame
    quat = imu.get_quaternion()                  # [4] wxyz
    joint_pos = motors.get_positions()            # [27] 按索引顺序
    joint_vel = motors.get_velocities()           # [27] 按索引顺序

    # 2. 计算重力投影
    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_gravity = quat_rotate_inverse(quat, gravity_world)

    # 3. 构造单帧观测 (90维)
    command = get_velocity_command()  # [vx, vy, wz]
    obs = np.concatenate([
        ang_vel,                               # [0:3]
        projected_gravity,                     # [3:6]
        command,                               # [6:9]
        joint_pos - default_joint_pos,         # [9:36]
        joint_vel,                             # [36:63]
        last_action,                           # [63:90]
    ])

    # 4. 推入历史 buffer, 拼接 10 帧
    obs_history.append(obs)
    obs_flat = np.concatenate(list(obs_history))  # [900]

    # 5. 策略推理
    with torch.no_grad():
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
        action = policy(obs_tensor).squeeze(0).numpy()

    # 6. 动作后处理
    action_clipped = np.clip(action, -100, 100)
    target_pos = action_clipped * ACTION_SCALE + default_joint_pos
    last_action = action_clipped

    # 7. 发送到电机
    motors.set_target_positions(target_pos)

    # 8. 保持 50Hz
    elapsed = time.time() - t0
    if elapsed < DT:
        time.sleep(DT - elapsed)
```

---

## 五、Sim-to-Real 差距及对策

| 差距来源 | 问题 | 对策 |
|---------|------|------|
| PD 参数 | 仿真 stiffness=9.156, damping=0.305；真机可能不同 | 实测阶跃响应，匹配仿真 |
| 通信延迟 | 仿真 0 延迟；真机总线 1-10ms | 仿真启用 action_delay 重新训练 |
| IMU 噪声 | 仿真高斯噪声；真机有漂移偏置 | 增大仿真 noise_scales |
| 关节摩擦 | 仿真无摩擦；真机舵机摩擦大 | 增大域随机化或 URDF 加摩擦 |
| 质量不准 | URDF ~1.2kg；真机含线缆电池 | 称量实际质量，修正 URDF |
| 脚底接触 | 仿真 box；真机形状材质不同 | 调整摩擦域随机化 |
| 控制频率 | 仿真精确 50Hz；真机可能抖动 | PREEMPT_RT 或稳定 50Hz |

---

## 六、渐进式调试流程

1. **验证单关节**：逐个关节发送角度、读取编码器，确认通信正确
2. **全关节同步**：27 关节同时控制，确认总线带宽和延迟
3. **静态站立**：发送 `default_joint_pos`，验证坐标系和关节映射正确性
4. **策略推理 + 悬空**：机器人挂起，运行策略观察腿部运动是否合理
5. **策略推理 + 落地**：放下机器人，逐步测试行走

---

## 七、缺失环节优先级

| 优先级 | 缺失环节 | 难度 | 说明 |
|--------|---------|------|------|
| P0 | FR02 硬件通信驱动 | 高 | 无此无法上真机 |
| P0 | 关节索引映射验证 | 中 | 仿真顺序 vs 真机总线 ID |
| P0 | IMU 驱动与坐标系校准 | 中 | body frame 必须一致 |
| P1 | 真机控制循环程序 | 中 | 参考上述伪代码或 LeggedLabDeploy |
| P2 | Sim-to-Real 参数匹配 | 高 | PD、延迟、摩擦调优 |
| P2 | 安全保护机制 | 中 | 跌倒检测、急停、力矩限制 |
| P3 | 带延迟/噪声重新训练 | 中 | 缩小 sim-to-real gap |

---

## 八、仿真侧推荐改进（上真机前）

1. 启用 `action_delay`：`self.domain_rand.action_delay.enable = True`
2. 增大域随机化范围
3. 增大 `noise_scales.joint_pos` 和 `joint_vel`
4. 训练 `fr02_rough` 任务增强泛化
5. 考虑 Sim-to-Sim（MuJoCo）跨引擎验证

---

## 九、参考资源

- LeggedLabDeploy：https://github.com/Hellod035/LeggedLabDeploy
- 观测构造源码：`legged_lab/envs/base/base_env.py` 第 130-184 行
- 动作后处理源码：`legged_lab/envs/base/base_env.py` 第 218-223 行
- 策略导出实现：`IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`
- FR02 资产配置：`legged_lab/assets/fr/fr.py`

---

*文档生成日期：2026-04-17*
