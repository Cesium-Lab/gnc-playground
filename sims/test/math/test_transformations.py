import sys
import numpy as np
import pytest

sys.path.append(".")

import csim.math.transformations as Trans
from csim.math.constants import DEG_TO_RAD
from csim.math.time import greg_to_jd

def test_lla_to_r_example_3_1():
    """Vallado 4e Example 3.1 (p. 141)"""
    lat_geodetic = 39.586667
    lon_geodetic = -105.640
    h_MSL = 4347.667
    r = Trans.surface_lla_to_ecef(lat_geodetic, lon_geodetic, h_MSL)

    r_horizontal = np.sqrt(r[0]**2 + r[1]**2)
    r_vert = r[2]

    # Book values are given in km
    assert r_horizontal == pytest.approx(4925.4298026 * 1e3) # m
    assert r_vert == pytest.approx(4045.4937426 * 1e3) # m
    assert np.arctan(r[1] / r[0]) == pytest.approx(DEG_TO_RAD * (lon_geodetic + 180))

def test_lla_to_r_example_3_2():
    """Vallado 4e Example 3.2 (p. 144)"""
    lat_geodetic = -(7 + 54/60 + 23.886/3600)
    lon_geodetic = 345 + 35/60 + 51/3600
    h_MSL = 56
    r = Trans.surface_lla_to_ecef(lat_geodetic, lon_geodetic, h_MSL)

    # Book values are given in km
    assert r[0] == pytest.approx(6119.400269 * 1e3)
    assert r[1] == pytest.approx(-1571.47955545 * 1e3)
    assert r[2] == pytest.approx(-871.56118090 * 1e3)

def test_r_to_lla_example_3_3():

    # Book values are given in km
    r = np.array([6524.834, 6862.875, 6448.296]) * 1e3

    lat, lon, alt = Trans.r_to_surface_lla(r)

    # Book values are given in km
    assert lat == pytest.approx(34.352496 * DEG_TO_RAD, rel=1e-5)
    assert lon == pytest.approx(46.4464 * DEG_TO_RAD)
    assert alt == pytest.approx(5085.22 * 1e3)

def test_vallado_itrf_W():
    """ Tests that determinant is 1. \\
        Vallado 4e Example 3-15 p. 230"""
    t_tt = 0.0426236319
    xp = -0.140682 * DEG_TO_RAD
    yp = 0.333309 * DEG_TO_RAD

    W = Trans.W_matrix(xp, yp, t_tt)

    assert np.linalg.det(W) == pytest.approx(1, 1e-9)

def test_vallado_itrf_R():
    """ Tests that determinant is 1. \\
        Vallado 4e Example 3-15 p. 230"""

    jd_ut1 = greg_to_jd(2004, 4, 6, 7, 51, 27.946047)
    
    R = Trans.R_matrix(jd_ut1)

    assert np.linalg.det(R) == pytest.approx(1, 1e-9)

def test_vallado_itrf_PN():
    """ Tests that determinant is 1. \\
        Vallado 4e Example 3-15 p. 230"""

# 0.042 623 631 9= = =

    t_tt = 0.0426236319
    dX = -0.000205
    dY = -0.000136
    
    PN = Trans.PN_matrix(t_tt, dX, dY)

    assert np.linalg.det(PN) == pytest.approx(1, 1e-9)

def test_vallado_full_itrf():
    # Book values are given in km
    r_itrf = np.array([-1033.479383, 7901.2952754, 6380.3565958]) * 1000

    jd = greg_to_jd(2004, 4, 6, 7, 51, 28.386009)

    xp = -0.140682 * DEG_TO_RAD
    yp = 0.333309 * DEG_TO_RAD
    dAT = 32
    dUT1 = 0.4399619
    dX = -0.000205
    dY = -0.000136

    PNRW = Trans.itrf_to_gcrs_matrix(xp, yp, jd, dAT, dUT1, dX, dY)

    assert np.linalg.det(PNRW) == pytest.approx(1, 1e-9)

    print(PNRW @ r_itrf)