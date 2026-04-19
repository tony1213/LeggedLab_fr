import numpy as np


def get_gravity_orientation(quaternion):
    """Compute projected gravity vector in body frame from quaternion [w,x,y,z]."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation
