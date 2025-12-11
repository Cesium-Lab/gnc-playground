import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

# =========================================================================================================== #
#                                                    Struct                                                   #
# =========================================================================================================== #

@dataclass
class Quat:
    v: np.ndarray

    @property
    def w(self):
        return self.v[0]

    @property
    def x(self):
        return self.v[1]

    @property
    def y(self):
        return self.v[2]

    @property
    def z(self):
        return self.v[3]


    @classmethod
    def from_vec(v: np.ndarray | list):
        v = np.asarray(v)
        assert v.shape == (4,)
        return Quat(v)
    
    @classmethod
    def I():
        return Quat(np.array([1,0,0,0]))
    
    @classmethod
    def wxyz(w: float, x: float, y: float, z: float):
        return Quat(np.array([w,x,y,z]))

# =========================================================================================================== #
#                                                   Angles                                                    #
# =========================================================================================================== #   

def unit(q: Quat):
    if abs(norm(q.v)) < 0.000001:
        q.v = np.zeros(4)
    else:
        q.v = q.v / norm(q.v)

    return q
# =========================================================================================================== #
#                                                   Angles                                                    #
# =========================================================================================================== #


def angle_between(v1: np.ndarray | list, v2: np.ndarray | list):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """

    if type(v1) is list:
        v1 = np.array([v1]).T
    if type(v2) is list:
        v2 = np.array([v2]).T


    if len(v1) == 3 and len(v2) == 3:

        num = np.dot(v1.T, v2)
        den = norm(v1) * norm(v2)

        result = np.arccos(num[0] / den)

        # breakpoint()

        return result

    elif len(v1) == 4 and len(v2) == 4:
        q_e = quat_mult(quat_inv(v1), v2)
        angle, _ = axis_rot_from_quat(q_e)
        return angle

    raise RuntimeError("Check lengths of input vectors")


# ----- Quaternion and Axis rotation -----#
def quat_from_axis_rot(angle_deg, axis):

    angle_rad = angle_deg * np.pi / 180.0
    axis_norm = axis / norm(axis)

    w = np.cos(angle_rad / 2)
    x, y, z = [np.sin(angle_rad / 2) * i for i in axis_norm]

    return np.array([w, x, y, z])


def axis_rot_from_quat(quat):
    w, x, y, z = quat

    angle = 2 * np.arccos(w)
    # print(f"{angle=}")

    if abs(angle) < 0.00000001:
        return 0, np.zeros(3)

    i = x / np.sin(angle / 2)
    j = y / np.sin(angle / 2)
    k = z / np.sin(angle / 2)

    return angle, np.array([i, j, k])


# ----- Quaternion and Rotation Matrix -----#
def R_from_quat(q):
    if len(q) != 4:
        print(f"Input quaternion should have 4 elements. Input was {q}")
        return np.identity(3)

    # 0 Quaternion
    if norm(q) == 0:

        return np.zeros((3, 3))

    q_norm = unit(q)

    # if abs(q_norm[0] - 1) < 1e-10:
    #     print(f"Identity quaternion with Q={q_norm}")

    w, i, j, k = q_norm

    R00 = 1 - 2 * (j * j + k * k)
    R01 = 2 * (i * j - k * w)
    R02 = 2 * (i * k + j * w)

    R10 = 2 * (i * j + k * w)
    R11 = 1 - 2 * (i * i + k * k)
    R12 = 2 * (j * k - i * w)

    R20 = 2 * (i * k - j * w)
    R21 = 2 * (j * k + i * w)
    R22 = 1 - 2 * (i * i + j * j)

    R = np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]])
    return R


def quat_from_R(R):

    # Makes sure the transpose can be taken
    if type(R) is not np.ndarray:
        R = np.array(R)

    assert np.shape(R) == (3,3)

    # Insomniac games formula
    row0, row1, row2 = R.T # Transpose because rows are columns in Insomniac convention
    m00, m01, m02 = row0
    m10, m11, m12 = row1
    m20, m21, m22 = row2

    if m22 < 0:
        if m00 > m11:
            t = 1 + m00 - m11 - m22
            q = np.array([t, m01 + m10, m20 + m02, m12 - m21])
        else:
            t = 1 - m00 + m11 - m22
            q = np.array([m01 + m10, t, m12 + m21, m20 - m02])
    else:
        if m00 < -m11:
            t = 1 - m00 - m11 + m22
            q = np.array([m20 + m02, m12 + m21, t, m01 - m10])
        else:
            t = 1 + m00 + m11 + m22
            q = np.array([m12 - m21, m20 - m02, m01 - m10, t])

    q *= 1 / 2 / np.sqrt(t)

    return np.array([q[3], q[0], q[1], q[2]])


# ----- Quaternion math! -----#
def quat_mult(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ----- Quaternion Operations -----#
def quat_inv(q):
    q = -q
    q[0] *= -1
    return q


def unit(q):
    if abs(norm(q)) < 0.000001:
        return np.zeros(len(q))

    return q / norm(q)


# ----- Applies Quaternion to Vector -----#
def quat_apply(quat, vector):
    quat = np.array(quat)
    temp = quat_mult(quat, [0, *vector])
    rslt = quat_mult(temp, quat_inv(quat))

    if abs(rslt[0]) > 0.0001:
        print(f"Quanternion is not normalized. Result vector of {rslt}")

    # Discards
    return rslt[1:4]


def cross_matrix(vec):

    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])