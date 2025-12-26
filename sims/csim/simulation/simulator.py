import numpy as np
from numpy.linalg import norm

from ..entities import Spacecraft
from ..math.integrators import rk4_func
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
from ..physics import grav_accel
from ..world import MU_EARTH
from ..physics.energy import calc_potential_energy, calc_kinetic_energy
# TODO: def finish this


class Simulator:
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float, spacecraft: Spacecraft):

        self.t = t0
        self.dt = dt
        self.n_steps = n_steps
        self.sc = spacecraft

        self.X = np.zeros((n_steps+1, 13))
        self.X[0] = state0

    def step_one(self, state: np.ndarray):
        """Return false if simulation is done"""

        # Simulate next state
        accel = grav_accel(state[:3])

        # # print(norm(accel))
        # mu = 3.986004418e14
        # r = state[:3]
        # accel = -mu * r / np.linalg.norm(r)**3

        sc = self.sc
        params = RigidBodyParams(sc.mass, sc.I, accel*sc.mass, np.zeros(3))

        next_state = rk4_func(self.t, self.dt, state, rigid_body_derivative, params)

        # next_state[3:6] = next_state[3:6] + accel * self.dt
        # next_state[:3] = state[:3] + next_state[3:6] * self.dt

        next_state[6:10] = next_state[6:10] / norm(next_state[6:10])
        
        return next_state   

    def simulate(self):

        self.final_step = self.n_steps
        
        for step in range(1,self.n_steps):
            
            # print(self.X[step-1])
            # prev_state = self.X[step-1]
            # print(calc_kinetic_energy(prev_state[3:6], prev_state[10:13], self.sc.mass, self.sc.I))
            # print(calc_potential_energy(prev_state[:3], self.sc.mass, MU_EARTH))
            next_step = self.step_one(self.X[step-1])
            self.t += self.dt

            self.X[step] = next_step

            # Vz
            # if step > 10 and self.X[step][2] <= 0:
            #     # breakpoint()
            #     self.final_step = step
            #     break


        self.X = self.X[:self.final_step]


    