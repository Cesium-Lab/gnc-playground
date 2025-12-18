# import sys
# sys.path.append(".")

# import numpy as np

# from csim.entities import Motor

# def test_initializer():
#     name = "test"
#     motor = Motor()

#     assert motor.name == "Motor"
#     assert not motor.loaded_thrust
#     assert motor.burn_time is None
#     assert motor.thrust is None
#     assert motor.time is None

#     motor = Motor(name)
#     assert motor.name == "test"
#     assert not motor.loaded_thrust
#     assert motor.burn_time is None
#     assert motor.thrust is None
#     assert motor.time is None

# def test_steps():
#     motor = Motor.thrust_from_step_throttle(
#         [1,2,3],
#         [3,4,5]
#     )

#     assert motor.burn_time == 12
#     assert np.array_equal(motor.thrust == [0,1,1,2,2])



    