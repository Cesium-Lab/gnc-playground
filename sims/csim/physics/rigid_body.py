#  ruff: noqa: E741
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..math.quaternion import hamilton_product
@dataclass
class RigidBodyParams:
    mass_kg: float
    I: np.ndarray
    force_N: np.ndarray
    torque_Nm: np.ndarray

def rigid_body_derivative(t: float, state: np.ndarray, params: RigidBodyParams):
    v = state[3:6]
    q = state[6:10]
    w = state[10:14]

    # Position derivative is velocity 
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = np.asarray(params.force_N) / params.mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    # print(q)
    # print(w)
    dqdt = 0.5 * hamilton_product(q, w)
    # print(dqdt)
    # Angular derivative based on (Schaub 4.34-35)
    I = params.I
    I_inv = np.linalg.inv(I)
    torque = np.asarray(params.torque_Nm)

    # τ = parameters.torque_body
    dwdt = I_inv @ (torque - np.cross(w, I @ w))

    return np.hstack((drdt, dvdt, dqdt, dwdt))
