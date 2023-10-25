# Functions to help with Trajectory Optimization
import numpy as np
from pydrake.systems.framework import VectorSystem


class QuaternionLQRStateSelector(VectorSystem):
    """
    A class for selecting state input for the Quaternion Linearization LQR.
    The scalar part of the quaternion is excluded (due to unit quaternion constraint)
    and hence has to be removed from the state feedback coming from the MultiBodyPlant block.
    This block removed the scalar part of quaternion and is placed between MultiBodyPlant and
    LQR blocks.
    """

    def __init__(self, numStates):

        VectorSystem.__init__(self, numStates, numStates - 1)

    def DoCalcVectorOutput(self, context, inputState, state_unused, outputState):
        """
        Exclude the 1st element of the state vector and return the rest
        """
        outputState[:] = inputState[1:]


def euler_to_quaternion(roll, pitch, yaw):
    """
    From:
    https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


def convert_w_to_quatdot(vars):
    """
    Betts Methods
    Function to convert angular velocity to 3x1 differential quaternion. q4 is used to change the
    length of the quaternion vector to unit vector in optimization.

    Input: 7x1 Numpy array: [quaternion, omega_dot]
    Output: 3x1 Numpy array: [quat1_dot, quat2_dot, quat3_dot]

    """
    assert vars.size == 7
    quat = vars[:4]
    w = vars[-3:]

    quat1_dot = 0.5 * ((w[0] * quat[3]) - (w[1] * quat[2]) + (w[2] * quat[1]))
    quat2_dot = 0.5 * ((w[0] * quat[2]) + (w[1] * quat[3]) - (w[2] * quat[0]))
    quat3_dot = 0.5 * (-(w[0] * quat[1]) + (w[1] * quat[0]) + (w[2] * quat[3]))
    quat_dot = np.array([quat1_dot, quat2_dot, quat3_dot])
    return quat_dot


def quat_multiply(q1, q2):
    """
    Multiply two quaternions in this order: q1(*)q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )
