"""
TODO:
- Normal J1 gravity
- J2, J3 even
- gravity gradient
"""
# ruff: noqa: E741

import numpy as np
from numpy.linalg import norm
from ..world import MU_EARTH, G_CONST, R_EARTH, J2_EARTH

def gravity(r: np.ndarray, mu: float = MU_EARTH):
    """Gravity acceleration \\
    Vallado 4e Equation 1-14 p. 23

    Args:
        r (np.ndarray): Position from center of large body [m/s]
        mu (float, optional): Gravitational parameter [m3/s2] Defaults to MU_EARTH.

    Returns:
        np.ndarray: Gravitational acceleration [m/s2]
    """
    
    g = -mu/(norm(r)**3) * r
    return g

def gravity_small_body(r: np.ndarray, mass_obj: float, mass_cb: float):
    """Gravity ***force*** with small body \\
    Vallado 4e Equation 1-13p. 23

    Args:
        r (np.ndarray): Position from center of body [m/s]
        mass_obj (float): Mass of object/sat/rocket [kg]
        mass_cb (float): Mass of central body [kg]

     Returns:
        np.ndarray: Force of gravity [N]
    """
    Fg = G_CONST * (mass_obj + mass_cb)/(norm(r)**3) * r
    return Fg


def gravity_J2(r: np.ndarray, r_cb: float = R_EARTH, mu: float = MU_EARTH):
    """Gravity J2 acceleration \\
    Could not for the life of me find in Vallado, but found in [Poliastro docs](https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html) and in [this paper](https://ntrs.nasa.gov/api/citations/20040031507/downloads/20040031507.pdf)

    Args:
        r (np.ndarray): Position from center of central body [m/s]
        r_cb (float, optional): Radius of central body [m]. Defaults to R_EARTH.
        mu (float, optional): Gravitational parameter [m3/s2] Defaults to MU_EARTH.

    Returns:
        np.ndarray: J2 perturbation acceleration [m/s2]
    """
    x,y,z = r
    r_norm = norm(r)
    coeff = 3/2 * J2_EARTH * mu * r_cb**2 / (2 * r_norm**5)
    axy = coeff * ((5*z*z)/(r_norm*r_norm) - 1)
    az = coeff * ((5*z*z)/(r_norm*r_norm) - 3)

    return np.array([axy*x, axy*y, az*z])

#TODO: test?
