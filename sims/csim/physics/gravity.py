"""
TODO:
- Norma; J1 gravity
- J2, J3 even
- gravity gradient
"""
# ruff: noqa: E741

import numpy as np
from numpy.linalg import norm
from ..world import MU_EARTH, G_CONST

def gravity(r: np.ndarray, mu: float = MU_EARTH):
    """Gravity force
    Vallado 4e Equation 1-14 p. 23

    Args:
        r (np.ndarray): Position from center of large body [m/s]
        mu (float, optional): Gravitational parameter [m3/s2] Defaults to MU_EARTH.

    Returns:
        np.ndarray: Force of gravity [m/s2]
    """
    Fg = -mu/(norm(r)**3) * r
    return Fg

def gravity_small_body(r: np.ndarray, mass_obj: float, mass_cb: float):
    """Gravity force with small body
    Vallado 4e Equation 1-13p. 23

    Args:
        r (np.ndarray): Position from center of body [m/s]
        mass_obj (float): Mass of object/sat/rocket [kg]
        mass_cb (float): Mass of central body [kg]

     Returns:
        np.ndarray: Force of gravity [m/s2]
    """
    Fg = G_CONST * (mass_obj + mass_cb)/(norm(r)**3) * r
    return Fg

#TODO: test?
