# ruff: noqa: E741
import numpy as np

from ..world.bodies import R_EARTH, R_EARTH_POLAR, ECC_EARTH
from .constants import DEG2RAD

def surface_lla_to_ecef(lat_deg: float, lon_deg: float, alt_m = 0):
    """From geodetic site coordinates to position vector \\
    Vallado 4e Example 3.1 p. 140

    Args:
        lat_deg (float): Geodetic latitude [rad]
        lon_deg (float): Geodetic longitude [rad]
        alt_m (int, optional): Altitude (MSL) [m]. Defaults to 0.

    Returns:
        np.ndarray: Position vector (ECEF) [m]
    """

    lat_gd = lat_deg * DEG2RAD
    lon_gd = lon_deg * DEG2RAD

    sin_lat = np.sin(lat_gd)
    cos_lat = np.cos(lat_gd)

    denom = np.sqrt(1 - ECC_EARTH**2 * sin_lat**2)
    C = R_EARTH / denom
    S = C * (1 - ECC_EARTH**2)

    # Site coordinates [km]
    r_horizontal = (C + alt_m)*cos_lat
    r_vertical = (S + alt_m)*sin_lat
    

    r = np.array([
        r_horizontal * np.cos(lon_gd),
        r_horizontal * np.sin(lon_gd),
        r_vertical
    ])

    return r

def r_to_surface_lla(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """Vallado 4e Algorithm 13 p. 173

    Args:
        r_ecef (np.ndarray): Position vector (ECEF) [m]

    Returns:
        tuple: Tuple with geodetic latitude [deg], longitude [deg], and ellipsoid altitude
    """
    ri, rj, rk = r_ecef

    r_horizontal_sat = np.sqrt(ri**2 + rj**2)
    a = R_EARTH

    # Vallado had this wrong :/
    # b = np.sqrt(R_pol*(1-ECC_EARTH**2))*np.sign(rk)
    b = R_EARTH_POLAR

    E = (b*rk - (a**2 - b**2)) / (a*r_horizontal_sat)

    lon = np.arcsin(rj/r_horizontal_sat)

    # Sanity check
    # print(lat)
    # print(np.arccos(ri/r_horizontal_sat))

    F = (b*rk + (a**2 - b**2)) / (a*r_horizontal_sat)
    P = 4*(E*F + 1) / 3
    Q = 2*(E**2 - F**2)
    D = P**3 - Q**2

    if D>0:
        v = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)
    else:
        v = 2*np.sqrt(-P)*np.cos(1/3*np.arccos(Q/(P*np.sqrt(-P))))

    G = 1/2*(np.sqrt(E**2 + v) + E)
    t = np.sqrt(G**2 + (F - v*G)/(2*G - E)) - G
    lat_gd = np.arctan(a*(1-t**2)/(2*b*t))
    h_ellp = (r_horizontal_sat - a*t)*np.cos(lat_gd) + (rk-b)*np.sin(lat_gd)
    
    return (lat_gd, lon, h_ellp)