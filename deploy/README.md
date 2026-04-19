# FR02 Real Robot Deployment

基于 Dynamixel Protocol 2.0 的 FR02 人形机器人策略部署工具。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 文件结构

```
deploy/
├── deploy_fr02.py          # 主控制脚本
├── configs/
│   └── fr02.yaml           # 机器人配置（关节映射、PD增益等）
├── common/
│   ├── __init__.py          # 重力投影计算
│   ├── dynamixel_driver.py  # Dynamixel Protocol 2.0 驱动
│   └── imu_driver.py        # IMU 抽象接口
├── policy/
│   └── fr02/
│       └── policy.pt        # 训练导出的 TorchScript 策略
└── requirements.txt
```

## 使用前必须完成的校准

### 1. Dynamixel 舵机 ID 映射

编辑 `configs/fr02.yaml` 中的 `joint2motor_id`，将每个仿真关节索引映射到对应的 Dynamixel 舵机 ID。

### 2. 零位偏移校准

编辑 `joint_zero_offset`：当 URDF 关节角度为 0 时，Dynamixel 舵机应处于的位置（弧度）。

校准方法：
1. 用 Dynamixel Wizard 2.0 将每个舵机移到 URDF 定义的零位
2. 读取此时的舵机位置值（ticks），转换为弧度
3. 填入 `joint_zero_offset` 对应位置

### 3. 方向校准

编辑 `joint_direction`：如果某个舵机正转方向与 URDF 定义相反，设为 -1。

### 4. IMU 驱动

编辑 `common/imu_driver.py` 中的 `SerialIMU._parse()` 方法，实现你具体 IMU 型号的数据解析。

## 运行

### Dry Run（无硬件测试）

```bash
cd deploy
python deploy_fr02.py --config configs/fr02.yaml --dry-run
```

### 真机运行

```bash
# 1. 确认 USB 设备权限
sudo chmod 666 /dev/ttyUSB0  # Dynamixel
sudo chmod 666 /dev/ttyUSB1  # IMU

# 2. 启动（默认前进 0.3 m/s）
python deploy_fr02.py --config configs/fr02.yaml --vx 0.3

# 3. 自定义速度指令
python deploy_fr02.py --config configs/fr02.yaml --vx 0.5 --vy 0.0 --wz 0.3
```

## 渐进式调试流程

1. **Dry Run**：`--dry-run` 验证策略加载和推理正常
2. **Ping 测试**：不带 `--dry-run` 启动，确认所有舵机上线
3. **默认位姿**：脚本自动移动到默认位姿，观察是否正确
4. **悬空测试**：将机器人挂起，按 ENTER 启动策略，观察腿部运动
5. **落地测试**：放下机器人

## 策略信息

| 项目 | 值 |
|------|-----|
| 输入维度 | 900 (90 x 10帧) |
| 输出维度 | 27 (关节动作) |
| 控制频率 | 50 Hz |
| Action Scale | 0.25 |
| 归一化 | 模型不含归一化，外部实现 |
