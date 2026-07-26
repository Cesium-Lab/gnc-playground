# Orbital Mechanics

## Two-body problem
- Orbits in a plane
- Specific energy is the same
  - Negative energy for in gravity well

### Classical Orbital Elements
- $a$ - semi major axis
- $e$ - eccentricity (elongation)
  - 0 - circular
  - 0-1 - ellipse
  - 1 - parabolic
  - \>1 - Hyperbolic
- $i$ - Inclination
  - 0-90º - Prograde
  - 90º-180º - retrograde
- $\Omega$ - Right ascension of ascending node (RAAN)
  - Where orbit crosses "zero" plane going up
- $\omega$ - Argument of periapsis
  - Periapsis location w.r.t RAAN
- $\theta$/$\nu$ - True anomaly
  - Also use $M$ for mean anomaly or $E$ for eccentric anomaly
- Degenerate cases
- Circular orbit
  - $e$ = 0
  - Periapsis is undefined
  - Argument of latitude used ($u = \omega + \nu$)
- Equatorial orbit
  - $i$ = 0
  - RAAN is undefined
  - Longitude of periapsis ($\bar{\omega} = \Omega + \omega$)
  - Basically just argument of periapsis w.r.t X axis of coords
- Circular AND equatorial
  - True longitude is used $l = \nu + \bar{\omega}$

![alt text](image-12.png)

**Modified Equinoctial Elements** (Just know they exist)
[Degenerate Conic article](https://degenerateconic.com/modified-equinoctial-elements.html)

**Kepler's Equation**
$M = E - e\ sin(E)$

**Vis-viva Equation**
[wikipedia](https://en.wikipedia.org/wiki/Vis-viva_equation)
Useful for $\Delta v$ checks

## Orbital Perturbations
There was a slideshow in AA 278 that I can't find. Bolded for important.
- [Spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics)
  - **[J2 (oblateness)](https://control.asu.edu/Classes/MAE462/462Lecture13.pdf)**
    - Earth's equatorial bulge causes secular (long-term, low) drift
  - Higher zonal harmonics (J3, J4, J5)
    - Orders of magnitude smaller than J2
  - Tesseral/sectoral
    - Why GEO needs stationkeeping too
    - ![alt text](image-13.png)
- **Third-body**
  - Sun, Moon, etc.
  - Relevant for cis-lunar 
    - ![alt text](image-9.png)
  - Negligible in LEO due to Earth gravity
- Tides
- Relativistic
- **Atmospheric drag**
  - Decay in $a$ and also depends on ballistic coefficient
- **Solar radiation pressure**
  - Reflected sunlight/radiation
- Albedo / IR radiation pressure
  - Same as SRP but from Earth or Moon
- Outgassing / thermal effects
  - Spacecraft's own materials/thrusters, notoriously affected early GPS orbit solutions before well modeled


Earth orbit
![alt text](image-11.png)

![alt text](image-14.png)

High altitude
![alt text](image-10.png)


With Moon Orbit
![alt text](image-15.png)



## TLE (two-line elements)
**Line 0**
![alt text](image-8.png)
- Satellite name

**Line 1**
![alt text](image-4.png)
![alt text](image-5.png)
- Time, designation

**Line 2**
![alt text](image-6.png)
![alt text](image-7.png)
- Orbital elements


### SGP4
- Simplified General Perturbation
  - Inputs of TLE

## Hill / Clohessy-Wiltshire equations
- Goal: describe a chaser's motion *relative* to a target on a circular reference orbit, without propagating full two-body dynamics for both bodies separately
- Derivation sketch: chaser position = target position + relative offset → substitute into the two-body EOM → linearize (first-order Taylor expansion) about the small relative offset → get linear, time-invariant equations in the rotating (Hill/LVLH) frame
- Result: in-plane (x, y) motion couples through the orbital rate n; out-of-plane (z) motion decouples into a simple harmonic oscillator
- Natural motion: in-plane relative orbits trace 2:1 ellipses around the target *unless* there's a net along-track drift term — always check for that drift term when reading a relative trajectory
- Key limitation: only valid for a circular (or near-circular) reference orbit and small relative separation. Eccentric references need the extended Tschauner-Hempel / Yamanaka-Ankersen formulations.

Links
- [Paper](https://sci-hub.ru/10.2514/8.8704)
- [Derivation (easy because they don't do it in the paper)](https://ensatellite.com/hills-equations/)

![alt text](image-3.png)

![alt text](image-17.png)
![alt text](image-16.png)
![alt text](image-18.png)


### Orbital maneuvers & propulsion
- **Hohmann transfer**: two-impulse, minimum-energy transfer between coplanar circular orbits — both Δv's come straight out of vis-viva
- **Bi-elliptic transfer**: beats Hohmann in total Δv for large orbit-radius ratios, at the cost of transfer time
- **Plane change**: Δv = 2v·sin(Δi/2) — expensive; cheapest when done at the lowest-velocity point of the orbit, ideally combined with another maneuver
- **Tsiolkovsky rocket equation**: Δv = Isp·g₀·ln(m₀/m_f) — connects propellant mass fraction to achievable Δv; know how to invert it to solve for required propellant given a Δv budget
### Perturbations
- **J2 (Earth oblateness)**: causes secular drift in RAAN and argument of periapsis — deliberately exploited for sun-synchronous orbit design
- **Atmospheric drag**: dominant in LEO, causes decay, scales with atmospheric density (varies with solar activity) and ballistic coefficient
- **Solar radiation pressure**: small but non-negligible for high area-to-mass spacecraft or long missions
- **Third-body effects**: sun/moon gravity perturbs Earth orbits, but becomes a *primary* force (not a perturbation) for cislunar trajectories — relevant given your Blue Origin lunar work
### Restricted three-body problem & Lagrange points
- CR3BP: two large primaries in circular orbits about their barycenter, a massless third body moving under both gravities
- 5 Lagrange points: L1/L2/L3 (collinear, unstable), L4/L5 (triangular, stable for suitable mass ratios)
- Halo/libration-point orbits around L1/L2 — periodic or quasi-periodic, relevant to cislunar/gateway-type missions. Worth knowing this exists and roughly why it's useful (low-energy transfers, persistent lunar-adjacent coverage) even without deriving it.
### Rigid body attitude dynamics
- **Euler's equations**: Iω̇ + ω × (Iω) = τ — nonlinear even with zero external torque, because angular velocity components couple through the inertia tensor
- Torque-free motion of an asymmetric body: rotation about the *intermediate* principal axis is unstable (the "tennis racket theorem") — a good one to have in your back pocket
- **Gravity gradient torque**: differential gravity across a body's extent creates a restoring torque toward local vertical — a passive attitude stabilization mechanism
- Angular momentum H = Iω is conserved (inertial frame, no external torque), even though ω itself isn't constant unless spinning about a principal axis

## Kepler's laws

## Lambert's problem

## Different orbits

## Trajectories TO moon and stuff

