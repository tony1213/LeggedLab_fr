"""Abstract IMU interface and serial-based implementation.

Replace or extend with your specific IMU driver.
"""

import time
from abc import ABC, abstractmethod

import numpy as np


class IMUBase(ABC):
    @abstractmethod
    def get_quaternion(self) -> np.ndarray:
        """Return orientation as [w, x, y, z] quaternion."""

    @abstractmethod
    def get_angular_velocity(self) -> np.ndarray:
        """Return angular velocity in body frame as [wx, wy, wz] (rad/s)."""

    @abstractmethod
    def close(self):
        pass


class DummyIMU(IMUBase):
    """Dummy IMU for testing without hardware. Reports standing upright."""

    def get_quaternion(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_angular_velocity(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def close(self):
        pass


class SerialIMU(IMUBase):
    """Generic serial IMU driver.

    TODO: Implement parsing for your specific IMU model.
    Most common IMUs (WT901, BNO055, HiPNUC, etc.) send binary/ASCII packets
    over serial. Adapt the parse method to your IMU's protocol.
    """

    def __init__(self, port: str, baudrate: int = 115200):
        import serial

        self.ser = serial.Serial(port, baudrate, timeout=0.01)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ang_vel = np.zeros(3, dtype=np.float32)
        time.sleep(0.5)
        print(f"[SerialIMU] Opened {port} @ {baudrate}bps")

    def update(self):
        """Read and parse latest IMU data from serial buffer.

        TODO: Replace with your IMU's actual parsing logic.
        """
        if self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            self._parse(data)

    def _parse(self, data: bytes):
        """Parse raw bytes into quaternion and angular velocity.

        TODO: Implement for your specific IMU protocol.
        """
        pass

    def get_quaternion(self) -> np.ndarray:
        self.update()
        return self._quat.copy()

    def get_angular_velocity(self) -> np.ndarray:
        self.update()
        return self._ang_vel.copy()

    def close(self):
        self.ser.close()
