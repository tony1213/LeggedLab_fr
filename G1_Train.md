# G1 机器人强化学习训练文档

> 本文档记录了使用 LeggedLab + Isaac Lab 框架对 Unitree G1 机器人进行强化学习训练的完整流程、命令、参数与经验总结。

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

## 二、任务说明

### 任务名称：`g1_plane`

在**纯平面地形（无起伏）** 上训练 G1 双足机器人行走，作为基准对比。

| 任务 | 地形 | 用途 |
|------|------|------|
| `g1_plane` | 纯平面（plane） | 基准训练，快速验证策略可行性 |
| `g1_flat` | 砾石地形（gravel） | 轻度不规则地面 |
| `g1_rough` | 粗糙地形（rough） | 强泛化能力训练，启用高度扫描 |

---

## 三、配置修改

训练前需要对 LeggedLab 做以下修改，以兼容 `rsl-rl >= 4.0.0`：

### 1. 新增 `g1_plane` 任务配置

**文件：`legged_lab/envs/g1/g1_config.py`**

```python
@configclass
class G1PlaneEnvCfg(G1FlatEnvCfg):
    """G1 训练：纯平面地形（无随机起伏），适合对比基准测试。"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None

@configclass
class G1PlaneAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "g1_plane"
    wandb_project: str = "g1_plane"
```

**文件：`legged_lab/envs/__init__.py`**（注册任务）

```python
from legged_lab.envs.g1.g1_config import (
    G1PlaneAgentCfg,
    G1PlaneEnvCfg,
    # ...
)
task_registry.register("g1_plane", BaseEnv, G1PlaneEnvCfg(), G1PlaneAgentCfg())
```

### 2. 兼容 rsl-rl >= 4.0.0

**文件：`legged_lab/scripts/train.py` 和 `play.py`**

```python
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

# 在 agent_cfg 构建后添加：
installed_version = metadata.version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
```

**文件：`legged_lab/envs/base/base_env_config.py`**（添加 obs_groups）

```python
@configclass
class BaseAgentCfg(RslRlOnPolicyRunnerCfg):
    # ...
    obs_groups: dict[str, list[str]] = {"actor": ["actor"], "critic": ["critic"]}
```

**文件：`legged_lab/envs/base/base_env.py`**（返回 TensorDict）

```python
from tensordict import TensorDict

def step(self, actions):
    # ...
    obs = TensorDict(
        {"actor": actor_obs, "critic": critic_obs},
        batch_size=(self.num_envs,),
        device=self.device,
    )
    return obs, reward_buf, self.reset_buf, self.extras

def get_observations(self):
    actor_obs, critic_obs = self.compute_observations()
    self.extras["observations"] = {"critic": critic_obs}
    return TensorDict(
        {"actor": actor_obs, "critic": critic_obs},
        batch_size=(self.num_envs,),
        device=self.device,
    )
```

---

## 四、训练命令

### 激活环境

```bash
source ~/miniconda3/bin/activate isaac
cd /home/rob/isaac_workspace/LeggedLab
export TERM=xterm
```

### 启动训练（后台运行，日志重定向）

```bash
nohup /home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/train.py \
  --task=g1_plane \
  --headless \
  --logger=tensorboard \
  --num_envs=2048 \
  --max_iterations=25000 \
  >> logs/train_g1_plane_25k.log 2>&1 &

echo "训练 PID: $!"
```

### 实时查看训练日志

```bash
tail -f logs/train_g1_plane_25k.log
```

### 查看 TensorBoard

```bash
source ~/miniconda3/bin/activate isaac
tensorboard --logdir logs/g1_plane
# 浏览器打开 http://localhost:6006
```

---

## 五、推理（Play）命令

### Headless 模式（无 GUI，用于快速验证）

```bash
source ~/miniconda3/bin/activate isaac
cd /home/rob/isaac_workspace/LeggedLab
export TERM=xterm

/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/play.py \
  --task=g1_plane \
  --headless \
  --num_envs=16 \
  --load_run=2026-04-15_14-35-00 \
  --checkpoint=model_24999.pt
```

### GUI 可视化模式（需先终止其他 Isaac 进程释放 VRAM）

```bash
# 确保没有其他 Isaac 进程占用 GPU
pkill -f 'play.py\|train.py'
sleep 3

source ~/miniconda3/bin/activate isaac
cd /home/rob/isaac_workspace/LeggedLab
export TERM=xterm DISPLAY=:1

/home/rob/isaac_workspace/IsaacLab/isaaclab.sh -p legged_lab/scripts/play.py \
  --task=g1_plane \
  --num_envs=16 \
  --load_run=2026-04-15_14-35-00 \
  --checkpoint=model_24999.pt
```

> **参数说明：**
> - `--load_run`：`logs/g1_plane/` 下的时间戳子目录名
> - `--checkpoint`：具体权重文件名（最终为 `model_24999.pt`）
> - `--num_envs`：推理时建议 16～50，无需 2048

---

## 六、训练结果（本次 g1_plane 25000 iter）

### 基本信息

| 项目 | 数值 |
|------|------|
| 训练日期 | 2026-04-15 |
| 迭代次数 | 25000 |
| 环境数量 | 2048 |
| 单步迭代时间 | ~0.65～0.74 s |
| 总训练时间 | **4 小时 34 分钟** |
| 大约吞吐量 | ~66,500 steps/s |

### 最终性能指标

| 指标 | 最终值 | 说明 |
|------|--------|------|
| **Mean Reward** | **26.67** | 综合奖励（越高越好） |
| **Mean Episode Length** | **1000** | 满帧未跌倒（最大值） |
| Mean Value Loss | 0.0089 | Critic 损失（接近 0 为好） |
| Mean Entropy Loss | 0.4934 | 策略熵（保持探索性） |
| Mean Action Std | 0.32 | 动作标准差（稳定后降低） |

### 分项奖励

| 奖励项 | 最终值 | 含义 |
|--------|--------|------|
| `track_lin_vel_xy_exp` | **0.9193** | 线速度跟踪（91.9%，接近满分） |
| `track_ang_vel_z_exp` | **0.8087** | 角速度跟踪（80.9%） |
| `termination_penalty` | **0.0000** | 零跌倒终止 |
| `feet_air_time` | 0.0061 | 迈步步态奖励 |
| `energy` | -0.0089 | 能量消耗（较低） |
| `action_rate_l2` | -0.1020 | 动作平滑性惩罚 |
| `flat_orientation_l2` | -0.0013 | 躯干保持水平 |
| `body_orientation_l2` | -0.0055 | 身体姿态惩罚 |
| `joint_deviation_arms` | -0.0963 | 手臂偏离默认姿态 |
| `joint_deviation_hip` | -0.0380 | 髋关节偏离 |
| `undesired_contacts` | -0.0011 | 非足部接触地面 |
| `feet_slide` | -0.0216 | 足部滑动 |

### 导出文件

```
logs/g1_plane/2026-04-15_14-35-00/
├── model_24999.pt              # 最终权重（PyTorch）
├── exported/
│   ├── policy.pt               # TorchScript JIT（部署用）
│   └── policy.onnx             # ONNX 格式（跨框架部署）
└── events.out.tfevents.*       # TensorBoard 日志
```

---

## 七、超参数配置

### PPO 算法超参数（`G1FlatAgentCfg` 默认值）

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps_per_env` | 24 | 每个环境每次更新采样步数 |
| `num_learning_epochs` | 5 | 每批数据训练轮数 |
| `num_mini_batches` | 4 | Mini-batch 数量 |
| `learning_rate` | 1e-3 | 初始学习率 |
| `schedule` | adaptive | 自适应学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.2 | PPO clip 参数 |
| `entropy_coef` | 0.005 | 熵正则系数 |
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `value_loss_coef` | 1.0 | Critic 损失系数 |

### 策略网络架构

```
Actor:  MLP [960 → 512 → 256 → 128 → 29]  (ELU 激活)
Critic: MLP [1010 → 512 → 256 → 128 → 1]  (ELU 激活)
观测历史长度: actor=10, critic=10 帧
```

### 仿真配置

| 参数 | 值 |
|------|-----|
| `dt` | 0.005 s（200 Hz 物理步） |
| `decimation` | 4（控制频率 50 Hz） |
| `max_episode_length_s` | 20 s（训练）/ 40 s（推理） |
| `env_spacing` | 2.5 m |
| `gpu_max_rigid_patch_count` | 327680 |

---

## 八、奖励函数说明

| 类别 | 奖励项 | 权重 | 作用 |
|------|--------|------|------|
| 任务目标 | `track_lin_vel_xy_exp` | +1.0 | 跟踪 XY 线速度指令 |
| 任务目标 | `track_ang_vel_z_exp` | +1.0 | 跟踪偏航角速度指令 |
| 安全惩罚 | `termination_penalty` | -200 | 跌倒终止重罚 |
| 步态质量 | `feet_air_time` | +0.15 | 鼓励双足交替迈步 |
| 能量效率 | `energy` | -1e-3 | 抑制高功耗动作 |
| 平滑性 | `action_rate_l2` | -0.01 | 抑制动作突变 |
| 姿态稳定 | `flat_orientation_l2` | -1.0 | 躯干保持水平 |
| 姿态稳定 | `body_orientation_l2` | -2.0 | 上身姿态惩罚 |
| 关节约束 | `dof_pos_limits` | -2.0 | 防止关节超限 |
| 接触安全 | `undesired_contacts` | -1.0 | 禁止非足部触地 |
| 足部安全 | `feet_slide` | -0.25 | 抑制足部滑动 |
| 足部安全 | `feet_force` | -3e-3 | 抑制足部过大冲击力 |
| 关节姿态 | `joint_deviation_hip` | -0.15 | 髋/肩/肘保持默认 |
| 关节姿态 | `joint_deviation_arms` | -0.2 | 腰/肩/腕保持默认 |
| 关节姿态 | `joint_deviation_legs` | -0.02 | 腿部关节保持默认 |

---

## 九、经验与踩坑记录

### 1. rsl-rl >= 4.0.0 兼容性

**问题**：`KeyError: 'class_name'` / `ValueError: obs_groups 不包含 actor 键`

**原因**：rsl-rl 4.x 要求：
- agent 配置使用 `actor`/`critic` 分离模型配置，而非旧版 `policy` 字段
- 环境返回的观测必须是 `TensorDict`，包含 `"actor"` 和 `"critic"` 键

**修复**：
1. 在 train.py/play.py 中调用 `handle_deprecated_rsl_rl_cfg()`
2. 在 `BaseAgentCfg` 中添加 `obs_groups`
3. 修改 `BaseEnv.step()` 和 `get_observations()` 返回 `TensorDict`

### 2. Play GUI 模式崩溃

**问题**：GUI 模式运行 play.py 报 `No physics scene created` / `AttributeError: 'NoneType'`

**原因**：在非 headless 模式下，Isaac Sim 使用不同的 experience file（`isaaclab.python.kit`），physics scene 初始化顺序与 headless 模式不同，导致创建 articulation view 时 physics_sim_view 为 None。

**修复**：推理验证时使用 `--headless` 模式；GUI 可视化需确保没有其他 Isaac 进程占用 GPU。

### 3. 僵尸 GPU 进程导致 OOM

**问题**：多次启动 play.py 后，GPU 内存接近耗尽（11 GB / 12 GB），新进程被系统杀死

**原因**：被 kill 掉的 Isaac Sim 进程有时不会立即释放 GPU 显存，残留进程持续占用 5+ GB

**修复**：
```bash
# 确认并清理所有残留 Isaac 进程
nvidia-smi   # 查看 GPU 占用进程
kill -9 <pid1> <pid2> ...
```

### 4. 训练进度监控

训练过程中可通过以下方式监控：

```bash
# 实时查看日志
tail -f logs/train_g1_plane_25k.log

# 提取关键指标（在新终端运行）
grep -E 'Mean reward:|Mean episode|ETA:|Iteration time:|Time elapsed:' \
  logs/train_g1_plane_25k.log | tail -20
```

### 5. 训练时间估算

| 环境数量 | 迭代速度 | 25000 次预计时间 |
|---------|---------|----------------|
| 2048 | ~0.65 s/iter | **~4.5 小时** |
| 4096 | ~1.2 s/iter | ~8 小时 |
| 1024 | ~0.35 s/iter | ~2.5 小时（精度略低）|

> RTX 5070（12 GB）上，2048 环境是精度与速度的最佳平衡点。

### 6. Checkpoint 策略

- 每 **100 次迭代**自动保存一个 checkpoint（`save_interval=100`）
- 训练完成后最终 checkpoint 为 `model_<max_iter-1>.pt`（如 `model_24999.pt`）
- 推理时 `--load_run` 填写 `logs/<task>/` 下的时间戳目录名

---

## 十、后续训练建议

1. **增加迭代次数**：50000 次迭代可进一步提升性能，预计约 9 小时
2. **粗糙地形训练（`g1_rough`）**：在 g1_plane 基础上，用 `g1_rough` 任务进行 fine-tune，启用高度扫描传感器增强泛化能力
3. **调整奖励权重**：若希望机器人跑得更快，可适当提高 `track_lin_vel_xy_exp` 权重（建议 1.5～2.0）
4. **启用域随机化**：增强 `physics_material` 和 `add_base_mass` 的随机范围，提升真实场景鲁棒性
5. **导入真实机器人测试**：使用导出的 `policy.pt` 通过 ROS2/SDK 部署到真实 G1

---

*文档生成日期：2026-04-15 | 训练 run ID：`2026-04-15_14-35-00`*
