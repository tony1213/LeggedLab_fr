"""Dynamixel Protocol 2.0 driver for FR02 humanoid robot.

Uses SyncRead/SyncWrite for efficient multi-servo communication.
Reference: https://emanual.robotis.com/docs/en/dxl/protocol2/
"""

import numpy as np
from dynamixel_sdk import (
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_POSITION_P_GAIN = 84
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_D_GAIN = 80
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126

OPERATING_MODE_POSITION = 3
OPERATING_MODE_EXTENDED = 4

POSITION_RESOLUTION = 4096
RAD_PER_TICK = (2 * np.pi) / POSITION_RESOLUTION
TICK_PER_RAD = POSITION_RESOLUTION / (2 * np.pi)

VELOCITY_UNIT = 0.229  # rev/min per unit
RPM_TO_RAD_PER_SEC = 2 * np.pi / 60.0


class DynamixelDriver:
    def __init__(self, device_port: str, baudrate: int, motor_ids: list[int],
                 protocol_version: float = 2.0):
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)

        self.port = PortHandler(device_port)
        self.packet = PacketHandler(protocol_version)

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port {device_port}")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")

        self._sync_read_pos = GroupSyncRead(self.port, self.packet,
                                            ADDR_PRESENT_POSITION, 4)
        self._sync_read_vel = GroupSyncRead(self.port, self.packet,
                                            ADDR_PRESENT_VELOCITY, 4)
        self._sync_write_pos = GroupSyncWrite(self.port, self.packet,
                                              ADDR_GOAL_POSITION, 4)

        for dxl_id in self.motor_ids:
            self._sync_read_pos.addParam(dxl_id)
            self._sync_read_vel.addParam(dxl_id)

        print(f"[DynamixelDriver] Opened {device_port} @ {baudrate}bps, "
              f"{self.num_motors} servos")

    def ping_all(self) -> list[bool]:
        results = []
        for dxl_id in self.motor_ids:
            _, comm, err = self.packet.ping(self.port, dxl_id)
            results.append(comm == COMM_SUCCESS and err == 0)
        return results

    def set_operating_mode(self, mode: int = OPERATING_MODE_EXTENDED):
        for dxl_id in self.motor_ids:
            self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_OPERATING_MODE, mode)

    def enable_torque(self, enable: bool = True):
        val = 1 if enable else 0
        for dxl_id in self.motor_ids:
            self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, val)

    def set_pid_gains(self, p_gains: list[int], i_gains: list[int] | None = None,
                      d_gains: list[int] | None = None):
        for i, dxl_id in enumerate(self.motor_ids):
            self.packet.write2ByteTxRx(self.port, dxl_id, ADDR_POSITION_P_GAIN, p_gains[i])
            if i_gains is not None:
                self.packet.write2ByteTxRx(self.port, dxl_id, ADDR_POSITION_I_GAIN, i_gains[i])
            if d_gains is not None:
                self.packet.write2ByteTxRx(self.port, dxl_id, ADDR_POSITION_D_GAIN, d_gains[i])

    def read_positions(self) -> np.ndarray:
        """Read present positions of all servos. Returns raw ticks as int32."""
        self._sync_read_pos.txRxPacket()
        positions = np.zeros(self.num_motors, dtype=np.int32)
        for i, dxl_id in enumerate(self.motor_ids):
            positions[i] = self._sync_read_pos.getData(
                dxl_id, ADDR_PRESENT_POSITION, 4)
        return positions

    def read_velocities(self) -> np.ndarray:
        """Read present velocities of all servos. Returns raw units as int32."""
        self._sync_read_vel.txRxPacket()
        velocities = np.zeros(self.num_motors, dtype=np.int32)
        for i, dxl_id in enumerate(self.motor_ids):
            raw = self._sync_read_vel.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
            if raw > 0x7FFFFFFF:
                raw -= 0x100000000
            velocities[i] = raw
        return velocities

    def write_positions(self, positions_ticks: np.ndarray):
        """Write goal positions to all servos. Input is raw ticks as int32."""
        self._sync_write_pos.clearParam()
        for i, dxl_id in enumerate(self.motor_ids):
            val = int(positions_ticks[i])
            data = [DXL_LOBYTE(DXL_LOWORD(val)), DXL_HIBYTE(DXL_LOWORD(val)),
                    DXL_LOBYTE(DXL_HIWORD(val)), DXL_HIBYTE(DXL_HIWORD(val))]
            self._sync_write_pos.addParam(dxl_id, data)
        self._sync_write_pos.txPacket()

    def close(self):
        self.enable_torque(False)
        self.port.closePort()

    @staticmethod
    def rad_to_ticks(rad: float) -> int:
        return int(rad * TICK_PER_RAD)

    @staticmethod
    def ticks_to_rad(ticks: int) -> float:
        return float(ticks) * RAD_PER_TICK

    @staticmethod
    def velocity_raw_to_rad_per_sec(raw: int) -> float:
        return float(raw) * VELOCITY_UNIT * RPM_TO_RAD_PER_SEC
