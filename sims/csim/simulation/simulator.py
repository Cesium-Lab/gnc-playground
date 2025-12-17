import numpy as np
from .spacecraft import Spacecraft
from ..math.integrators import rk4_func
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams

# TODO: def finish this
class Simulator:
    def __init__(self, t0: float, dt: float, n_steps: float, spacecraft: Spacecraft):

        self.t_arr = np.linspace(t0, dt*n_steps, n_steps)
        self.t = t0
        self.dt = dt
        self.n_steps = n_steps
        self.sc = spacecraft

        self.X = np.zeros((n_steps+1, 13))
        self.X[0] = self.sc.state

    def step_one(self, step: int):
        """Return false if simulation is done"""
        
        force = np.array([0, 0, self.sc.thrust(self.t)]) + np.array([0,0,-9.81*self.sc.mass])

        sc = self.sc
        params = RigidBodyParams(sc.mass, sc.I, force, np.zeros(3))

        next_state = rk4_func(self.t, self.dt, self.sc.state, rigid_body_derivative, params)
        self.sc.state = next_state
        # print(next_state)
        self.X[step+1] = next_state

    def simulate(self):

        self.final_step = self.n_steps
        for step in range(1,self.n_steps):
            
            self.t += self.dt
            self.step_one(step)

            # Vz
            if step > 10 and self.sc.state[2] <= 0:
                # breakpoint()
                self.final_step = step
                break


        self.X = self.X[:self.final_step]


    