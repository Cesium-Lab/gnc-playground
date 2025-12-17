import numpy as np

class E:
    def __init__(self, v):
        self.v = v


arr = np.array([4])

a = E(arr)

print(a.v)

arr[0] += 1

print(a.v)
