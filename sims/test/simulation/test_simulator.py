# ruff: noqa: E741
import sys
import numpy as np
import pytest
import matplotlib.pyplot as plt

sys.path.append(".")

from csim.simulation import Simulator
from csim.entities import Spacecraft

t0 = 0
dt = 0.01
n_steps = 10000

state0 = np.array([0,0,0, 0,0,0, 0,0,0,0, 0,0,0])

sc = Spacecraft(100, np.identity(3), state0)
sim = Simulator(t0, dt, n_steps, sc)

sim.simulate()

print(sim.X.shape)
plt.plot(sim.X[:,:3])
plt.plot(sim.X[:,:6])
plt.show()