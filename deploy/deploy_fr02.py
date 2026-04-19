"""FR02 real-robot deployment controller.

Loads a trained policy (TorchScript JIT) and runs a 50Hz control loop
communicating with Dynamixel servos via Protocol 2.0.

Usage:
    python deploy_fr02.py --config configs/fr02.yaml
    python deploy_fr02.py --config configs/fr02.yaml --dry-run   # no hardware
"""

import argparse
import signal
import sys
import time

import numpy as np
import torch
import yaml

from common import get_gravity_orientation
from common.dynamixel_driver import DynamixelDriver
from common.imu_driver import DummyIMU, SerialIMU


class FR02Config:
    def __init__(self, path: str):
        with open(path) as f:
            cfg = yaml.safe_load(f)

        self.control_dt = cfg["control_dt"]
        self.policy_path = cfg["policy_path"]
        self.history_length = cfg["history_length"]
        self.num_actions = cfg["num_actions"]
        self.num_obs = cfg["num_obs"]

        self.dynamixel = cfg["dynamixel"]
        self.imu_cfg = cfg["imu"]

        self.joint2motor_id = cfg["joint2motor_id"]
        self.joint_zero_offset = np.array(cfg["joint_zero_offset"], dtype=np.float64)
        self.joint_direction = np.array(cfg["joint_direction"], dtype=np.float64)
        self.default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

        self.kps = cfg["kps"]
        self.kds = cfg["kds"]

        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.command_scale = np.array(cfg["command_scale"], dtype=np.float32)

        cmd = cfg["command_range"]
        self.clip_min_cmd = np.array([cmd["lin_vel_x"][0], cmd["lin_vel_y"][0],
                                      cmd["ang_vel_z"][0]], dtype=np.float32)
        self.clip_max_cmd = np.array([cmd["lin_vel_x"][1], cmd["lin_vel_y"][1],
                                      cmd["ang_vel_z"][1]], dtype=np.float32)


class FR02Controller:
    def __init__(self, config: FR02Config, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.running = True

        self.policy = torch.jit.load(config.policy_path).eval()

        n = config.num_actions
        self.action = np.zeros(n, dtype=np.float32)
        self.joint_pos = np.zeros(n, dtype=np.float32)
        self.joint_vel = np.zeros(n, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.obs_history = np.zeros((config.history_length, config.num_obs),
                                    dtype=np.float32)
        self.first_run = True

        for _ in range(50):
            with torch.inference_mode():
                obs = self.obs_history.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs))
        print("[Policy] Warmup complete")

        if dry_run:
            print("[DRY RUN] No hardware will be accessed")
            self.dxl = None
            self.imu = DummyIMU()
        else:
            self.dxl = DynamixelDriver(
                device_port=config.dynamixel["device_port"],
                baudrate=config.dynamixel["baudrate"],
                motor_ids=config.joint2motor_id,
                protocol_version=config.dynamixel["protocol_version"],
            )
            alive = self.dxl.ping_all()
            for i, ok in enumerate(alive):
                if not ok:
                    print(f"  [WARN] Servo ID {config.joint2motor_id[i]} not responding")
            print(f"[Dynamixel] {sum(alive)}/{len(alive)} servos online")

            self.dxl.enable_torque(False)
            self.dxl.set_operating_mode()
            self.dxl.set_pid_gains(config.kps, d_gains=config.kds)
            self.dxl.enable_torque(True)

            if config.imu_cfg["type"] == "serial":
                self.imu = SerialIMU(config.imu_cfg["device_port"],
                                     config.imu_cfg["baudrate"])
            else:
                self.imu = DummyIMU()
                print("[IMU] Using dummy IMU (no real IMU configured)")

    def sim_rad_to_dxl_rad(self, sim_rad: np.ndarray) -> np.ndarray:
        """Convert simulation joint angles (rad) to Dynamixel target (rad)."""
        return self.cfg.joint_zero_offset + self.cfg.joint_direction * sim_rad

    def dxl_rad_to_sim_rad(self, dxl_rad: np.ndarray) -> np.ndarray:
        """Convert Dynamixel present position (rad) to simulation joint angle (rad)."""
        return self.cfg.joint_direction * (dxl_rad - self.cfg.joint_zero_offset)

    def read_joint_state(self):
        """Read joint positions and velocities from Dynamixel servos."""
        if self.dxl is None:
            return

        raw_pos = self.dxl.read_positions()
        raw_vel = self.dxl.read_velocities()

        for i in range(self.cfg.num_actions):
            dxl_rad = DynamixelDriver.ticks_to_rad(raw_pos[i])
            self.joint_pos[i] = (self.cfg.joint_direction[i]
                                 * (dxl_rad - self.cfg.joint_zero_offset[i]))
            self.joint_vel[i] = (self.cfg.joint_direction[i]
                                 * DynamixelDriver.velocity_raw_to_rad_per_sec(raw_vel[i]))

    def move_to_default_pos(self, duration: float = 3.0):
        """Smoothly interpolate from current position to default position."""
        if self.dxl is None:
            print("[DRY RUN] Would move to default position")
            return

        self.read_joint_state()
        init_pos = self.joint_pos.copy()
        steps = int(duration / self.cfg.control_dt)

        print(f"Moving to default position over {duration}s...")
        for i in range(steps):
            alpha = (i + 1) / steps
            target_sim = init_pos * (1 - alpha) + self.cfg.default_joint_pos * alpha
            target_dxl_rad = self.sim_rad_to_dxl_rad(target_sim)
            target_ticks = np.array([DynamixelDriver.rad_to_ticks(r)
                                     for r in target_dxl_rad], dtype=np.int32)
            self.dxl.write_positions(target_ticks)
            time.sleep(self.cfg.control_dt)
        print("Default position reached.")

    def set_command(self, vx: float, vy: float, wz: float):
        """Set velocity command (m/s, m/s, rad/s)."""
        cmd = np.array([vx, vy, wz], dtype=np.float32) * self.cfg.command_scale
        self.command = np.clip(cmd, self.cfg.clip_min_cmd, self.cfg.clip_max_cmd)

    def step(self):
        """Execute one control step: read sensors -> infer -> write actuators."""
        self.read_joint_state()

        quat = self.imu.get_quaternion()
        ang_vel = self.imu.get_angular_velocity()

        gravity = get_gravity_orientation(quat)

        n = self.cfg.num_actions
        obs = np.zeros(self.cfg.num_obs, dtype=np.float32)
        obs[0:3] = ang_vel * self.cfg.ang_vel_scale
        obs[3:6] = gravity
        obs[6:9] = self.command
        obs[9:9 + n] = (self.joint_pos - self.cfg.default_joint_pos) * self.cfg.dof_pos_scale
        obs[9 + n:9 + 2 * n] = self.joint_vel * self.cfg.dof_vel_scale
        obs[9 + 2 * n:9 + 3 * n] = self.action

        if self.first_run:
            self.obs_history[:] = obs.reshape(1, -1)
            self.first_run = False
        else:
            self.obs_history = np.concatenate(
                (self.obs_history[1:], obs.reshape(1, -1)), axis=0)

        obs_flat = self.obs_history.reshape(1, -1).astype(np.float32)

        with torch.inference_mode():
            action_tensor = self.policy(
                torch.from_numpy(obs_flat).clip(-100, 100)
            ).clip(-100, 100)
            self.action = action_tensor.detach().numpy().squeeze()

        target_sim = self.cfg.default_joint_pos + self.action * self.cfg.action_scale

        if self.dxl is not None:
            target_dxl_rad = self.sim_rad_to_dxl_rad(target_sim)
            target_ticks = np.array([DynamixelDriver.rad_to_ticks(r)
                                     for r in target_dxl_rad], dtype=np.int32)
            self.dxl.write_positions(target_ticks)

        return target_sim

    def run(self):
        """Main control loop at 50Hz."""
        print("\n=== FR02 Policy Control Loop ===")
        print(f"  Control frequency: {1.0 / self.cfg.control_dt:.0f} Hz")
        print(f"  History length: {self.cfg.history_length}")
        print(f"  Action scale: {self.cfg.action_scale}")
        print(f"  Dry run: {self.dry_run}")
        print("Press Ctrl+C to stop.\n")

        step_count = 0
        t_start = time.time()

        try:
            while self.running:
                t0 = time.perf_counter()

                target = self.step()
                step_count += 1

                if step_count % 50 == 0:
                    elapsed = time.time() - t_start
                    actual_hz = step_count / elapsed if elapsed > 0 else 0
                    print(f"[Step {step_count:>6d}] "
                          f"freq={actual_hz:.1f}Hz "
                          f"cmd=[{self.command[0]:.2f},{self.command[1]:.2f},{self.command[2]:.2f}] "
                          f"action_norm={np.linalg.norm(self.action):.3f}")

                dt = time.perf_counter() - t0
                sleep_time = self.cfg.control_dt - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")

        self.shutdown()

    def shutdown(self):
        self.running = False
        if self.dxl is not None:
            print("Disabling torque...")
            self.dxl.enable_torque(False)
            self.dxl.close()
        self.imu.close()
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="FR02 Real Robot Deployment")
    parser.add_argument("--config", type=str, default="configs/fr02.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without hardware (dummy sensors)")
    parser.add_argument("--vx", type=float, default=0.3,
                        help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    args = parser.parse_args()

    config = FR02Config(args.config)
    controller = FR02Controller(config, dry_run=args.dry_run)

    signal.signal(signal.SIGINT, lambda s, f: setattr(controller, 'running', False))

    controller.move_to_default_pos()

    input("Press ENTER to start policy control (or Ctrl+C to abort)...")

    controller.set_command(args.vx, args.vy, args.wz)
    controller.run()


if __name__ == "__main__":
    main()
