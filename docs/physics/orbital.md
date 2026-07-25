### Two-body problem & Kepler's laws
- Governing equation: r̈ = −(μ/r³)·r (central inverse-square gravity, no other forces)
- Conserved quantities: specific angular momentum h = r × v (constant, defines the orbit plane and shape), specific mechanical energy ε = v²/2 − μ/r (constant, sign tells you the orbit type)
- ε < 0 → ellipse, ε = 0 → parabola, ε > 0 → hyperbola
- Classical orbital elements: a, e, i, Ω, ω, ν — six numbers, equivalent information to a Cartesian state vector, just a different parameterization
- **Vis-viva equation**: v² = μ(2/r − 1/a) — fast way to get speed at any point in an orbit from position and semi-major axis, useful for quick delta-v sanity checks
- **Kepler's equation**: M = E − e·sin(E) (M = mean anomaly, linear in time; E = eccentric anomaly). Transcendental — solved iteratively (Newton-Raphson). It's the bridge between "how much time has passed" and "where are you in the orbit" for anything eccentric.
### Hill / Clohessy-Wiltshire equations
- Goal: describe a chaser's motion *relative* to a target on a circular reference orbit, without propagating full two-body dynamics for both bodies separately
- Derivation sketch: chaser position = target position + relative offset → substitute into the two-body EOM → linearize (first-order Taylor expansion) about the small relative offset → get linear, time-invariant equations in the rotating (Hill/LVLH) frame
- Result: in-plane (x, y) motion couples through the orbital rate n; out-of-plane (z) motion decouples into a simple harmonic oscillator
- Natural motion: in-plane relative orbits trace 2:1 ellipses around the target *unless* there's a net along-track drift term — always check for that drift term when reading a relative trajectory
- Key limitation: only valid for a circular (or near-circular) reference orbit and small relative separation. Eccentric references need the extended Tschauner-Hempel / Yamanaka-Ankersen formulations.
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

