import sys
import numpy as np
import pytest
from numpy.linalg import norm
from pprint import pprint
sys.path.append(".")

from csim.math.coes import kelper_eq_ellipse, rv_to_coes
from csim.math.constants import DEG_TO_RAD, RAD_TO_DEG
from csim.world.bodies import MU_EARTH

def test_kelper():
    """Vallado 4e Example 2-1 p. 66"""
    M = 235.4 * DEG_TO_RAD
    e = 0.4
    E = kelper_eq_ellipse(M, e)

    assert E * RAD_TO_DEG == pytest.approx(220.512074767522)

class TestRvToCoes:

    def test_vallado(self):
        """Vallado 4e p. 115"""
        r = np.array([6524.834, 6862.875, 6448.296]) # km
        v = np.array([4.901327, 5.533756, -1.976341]) # km/s

        a,e,i,raan,aop,ta = rv_to_coes(r,v)

        assert a == pytest.approx(36127.343, rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)

        """Vallado 4e p. 115 but in meters"""
        r = np.array([6524.834, 6862.875, 6448.296]) * 1000 # m
        v = np.array([4.901327, 5.533756, -1.976341]) * 1000 # m/s

        a,e,i,raan,aop,ta = rv_to_coes(r,v, MU_EARTH)

        assert a == pytest.approx(36127343., rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)

    def test_random(self):
        """ http://orbitsimulator.com/formulas/OrbitalElements.html """

        r = np.array([100, 1300, -6000]) # km
        v = np.array([0.1, -1.0, 2.1]) # km/s

        a,e,i,raan,aop,ta = rv_to_coes(r,v)

        assert a == pytest.approx(3203.754267903962, rel=1e-6)
        assert e == pytest.approx(0.9955257908191099, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(93.90569432920545, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(283.91249467651966, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(77.26657593515651, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(181.10299292301673, rel=1e-5)

    def test_circular(self):
        """ http://orbitsimulator.com/formulas/OrbitalElements.html """

        r = np.array([6000, 4713.675910949817, 1000]) # km
        v = np.array([4.446, -5.658, 0]) # km/s (circular orbit)

        a,e,i,raan,_,_ = rv_to_coes(r,v)

        assert a == pytest.approx(7692.648, rel=1e-4)
        assert e == pytest.approx(0, abs=1e-3)
        assert i * RAD_TO_DEG == pytest.approx(172.53, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(128.1599, rel=1e-5)
        # argument of perigee and true anomaly are undefined for circular orbits (can be any value)

    def test_0_inclination(self):
        """ http://orbitsimulator.com/formulas/OrbitalElements.html """

        r = np.array([6800, 0, 0]) # km
        v = np.array([0, 8.656278815328648, 0]) # km/s

        a,e,i,_,_,_ = rv_to_coes(r,v)

        assert a == pytest.approx(9421.974578532005, rel=1e-4)
        assert e == pytest.approx(0.2782829179465395, abs=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(0, rel=1e-4)
        # raan, aop and ta are undefined for circular orbits (can be any value)

    def test_90_inclination(self):
        """ http://orbitsimulator.com/formulas/OrbitalElements.html """

        r = np.array([6686, 0, 968.35]) # km
        v = np.array([0, 0, -5.899127972862638]) # km/s

        a,e,i,raan,aop,ta = rv_to_coes(r,v)

        assert a == pytest.approx(4790.6431, rel=1e-5)
        assert e == pytest.approx(0.4305248975978197, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(90, rel=1e-5)
        assert raan * RAD_TO_DEG == pytest.approx(180, rel=1e-5) # Since going down initially, ascends on other side
        assert aop * RAD_TO_DEG == pytest.approx(340.552, rel=1e-5)
        assert ta * RAD_TO_DEG == pytest.approx(191.206, rel=1e-5)

    def test_180_inclination(self):
        """ http://orbitsimulator.com/formulas/OrbitalElements.html """

        r = np.array([7000, 2, 0]) # km
        v = np.array([1.2, -2, 0]) # km/s

        a,e,i,_,_,_ = rv_to_coes(r,v)

        assert a == pytest.approx(3675.56885, rel=1e-5)
        assert e == pytest.approx(0.930685, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(180, rel=1e-5)
        # raan, aop, and ta are undefined for equitorial orbits (can be any value)

class TestCoesToEv:
    pass

#     @testset "coes to rv" begin
#         @testset "Vallado Example" begin
#             # Vallado 4th Edition, pg 119

#             coes = OrbitalElements(
#                 36126.64,
#                 0.83285,
#                 87.87 * DEG_TO_RAD,
#                 227.89 * DEG_TO_RAD,
#                 53.38 * DEG_TO_RAD,
#                 92.335 * DEG_TO_RAD,
#             )

#             r,v = orbital_elements_to_rv(coes, μ)

#             @test isapprox(r, [6525.344, 6861.535, 6449.125], atol=0.1)
#             @test isapprox(v, [4.902276, 5.533124, -1.975709], atol=0.0001)
#         end
        
#         @testset "Random Example" begin
#             """ http://orbitsimulator.com/formulas/OrbitalElements.html """
#             coes = OrbitalElements(
#                 3203.754267903962,
#                 0.9955257908191099,
#                 93.90569432920545 * DEG_TO_RAD,
#                 283.91249467651966 * DEG_TO_RAD,
#                 77.26657593515651 * DEG_TO_RAD,
#                 181.10299292301673 * DEG_TO_RAD,
#             )

#             r,v = orbital_elements_to_rv(coes, μ)

#             @test isapprox(r, [100.0, 1300.0, -6000.0], atol=0.00001)
#             @test isapprox(v, [0.1, -1.0, 2.1], atol=0.0001)
#         end

#         @testset "90 Inclination" begin
#             """ http://orbitsimulator.com/formulas/OrbitalElements.html """
#             coes = OrbitalElements(
#                 4848.9197268518665,
#                 0.3938736352565609,
#                 90.0 * DEG_TO_RAD,
#                 180.0 * DEG_TO_RAD,
#                 353.8847334059262 * DEG_TO_RAD,
#                 177.87427980631205 * DEG_TO_RAD,
#             )

#             r,v = orbital_elements_to_rv(coes, μ)

#             @test isapprox(r, [6686, 0.0, 968.35], atol=1e-6)
#             @test isapprox(v, [1.0, 0.0, -5.899127972862638], atol=1e-4)
#         end

#         # Not testing Circular Equatorial, Circular Inclined, and Elliptical Equatorial cases
#         # we'd realistically only be using the rv_to_coes function since the source of truth
#         # for position are our r and v vectors

#     end

# end

# @testset "Orbital Element Time Tests" begin

#     @testset "Mean Motion to Semi-Major Axis" begin
#         # ISS data
#         # (not much of a validation, since there is no semi-major axis in the TLE, but a sanity-check for ballpark)
#         @test isapprox(mean_motion_to_sma(15.4978258, μ), 7071.85952, atol=.001)
#     end
#     @testset "Orbital Period to Semi-Major Axis" begin
#         # Vallado 4th edition pg. 31
#         @test isapprox(orbit_period_to_sma(86164.090518, μ), 42164.1696, atol=.001)

#         period = 92.9 * 60 # ISS orbital period [s]
#         sma = mean_motion_to_sma(15.4978258, μ) # semi-major axis [km]
#         # @test isapprox(orbit_period_to_sma(period, μ), sma, atol=.1)
#     end
    
#     @testset "Semi-Major Axis to Orbital Period" begin
#         # Vallado 4th edition pg. 31
#         @test isapprox(sma_to_orbit_period(42164.1696, μ), 86164.090518, atol=.001)

#         period = 92.9 * 60 # ISS orbital period [s]
#         sma = mean_motion_to_sma(15.4978258, μ) # semi-major axis [km]
#         # @test isapprox(sma_to_orbit_period(sma, μ), period, atol=.001)
#     end

# # TODO: implement these to check the functions
#     """
# | Orbit Type                                    | Altitude (km) | Semi-Major Axis a = Rₑ + h (km) | Period T (s) | Period (min)     | Source / Notes                                         |
# | --------------------------------------------- | ------------- | ------------------------------- | ------------ | ---------------- | ------------------------------------------------------ |
# | **ISS / LEO**                                 | 408           | 6779                            | 5549.8       | 92.5             | NASA ISS Fact Sheet                                    |
# | **Sun-Synchronous (typical)**                 | 720           | 7091                            | 5973.0       | 99.6             | Common for Earth observation (e.g., Landsat, Sentinel) |
# | **Polar LEO (alt. 1000 km)**                  | 1000          | 7371                            | 6283.2       | 104.7            | General reference for high LEO                         |
# | **MEO (GPS)**                                 | 20200         | 26571                           | 43082.0      | 718.0 (≈11.97 h) | GPS constellation standard                             |
# | **Molniya (apogee 39750 km, perigee 600 km)** | —             | 26600 (mean)                    | 43000        | 716.7            | High-eccentricity 12h orbit                            |
# | **GEO (Geostationary)**                       | 35786         | 42157                           | 86164.1      | 1436.1 (23h56m)  | Standard GEO, 1 sidereal day period                    |

#     """
    