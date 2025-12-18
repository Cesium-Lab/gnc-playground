import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ...constants import LBF_TO_N, N_TO_LBF

class Motor:

    def __init__(self, name = "Motor"):
        """Ideally should not be calling this method. 
        Use `Motor.from_csv()` or `Motor.from_step()` or `Motor.from_rse()`"""
        self.name = name

        self.loaded_thrust = False
        self.burn_time = None
        self.thrust = None
        self.time = None
        # self.total_impulse = None
        # self.impulse = None
        print(f"Creating motor with name '{name}'")

       
    @staticmethod
    def thrust_from_step_throttle(thrusts, times, name = "Motor", thrust_units = "lbf"):

        # Unit
        if thrust_units == 'lbf':
            print("Setting default thrust units to lbf")
            unit_scale = LBF_TO_N
        elif thrust_units == 'N':
            print("Setting thrust units to Newtons")
            unit_scale = 1.0
        else:
            print("Setting thrust units to Newtons by default")
            unit_scale = 1.0
        
        motor = Motor(name)

        motor.thrust = [0]
        motor.time = [0]
        t_curr = 0

        for T,t in zip(thrusts, times):

            motor.thrust.append([T,T])
            motor.time.append([t_curr+0.00001, t_curr + t])
            t_curr += t

        motor.burn_time = t_curr
        assert motor.burn_time == sum(times)


        # motor.time = np.array([0.0, time[0], time[0] + 0.00001, time[1]])
        # motor.thrust = np.array([thrust[0], thrust[0], thrust[1], thrust[1]]) * unit_scale
        # motor.burn_time = max(time)

   
      
        # motor.I_tot = np.trapz(y=motor.thrust, x=motor.time)

        motor.loaded_thrust = True

        return motor



    # def thrust_from_csv(self, filepath, delimeter = ",", column_names = None, thrust_units = None):


    #     data = pd.read_csv(filepath, delimiter=delimeter)

    #     unit_conversion = 1.0

    #     if column_names:
    #         time_col = column_names[0]
    #         thrust_col = column_names[1]
            

    #     else:
    #         time_col = data.columns[0]
    #         thrust_col = data.columns[1]

    #     if thrust_units == 'lbf':
    #         print("Overriding thrust units to lbf")
    #         unit_conversion = LBF2N
            
    #     if '(lbf)' in thrust_col:
    #         print("Auto-detected thrust in units of lbf")
    #         unit_conversion = LBF2N
        
    #     self.thrust = data[thrust_col] * unit_conversion
    #     self.time = data[time_col]
    #     self.tb = max(self.time)

    #     self.I_tot = np.trapz(y=self.thrust, x=self.time)

    #     self.loaded_thrust = True


 
    # def thrust_from_rse(self, filepath):

    #     force_arr = []
    #     time_arr = []
    #     # mass_arr = []
    #     with open(filepath) as rse_file:
    #         for line in rse_file:

    #             if '<eng-data' not in line:
    #                 continue

    #             line = line.split('"')

    #             force = float(line[3])
    #             # mass = float(line[5])
    #             time = float(line[7])

    #             force_arr.append(force)
    #             time_arr.append(time)
    #             # mass_arr.append(mass)

    #             # print(line)
    #             # print(f"{time}s, {force * N2LBF} LBF")
                
        
    #     self.thrust = np.array(force_arr)
    #     self.time = np.array(time_arr)

    #     self.I_tot = np.trapz(y=self.thrust, x=self.time)

    #     print(self.I_tot)

    #     self.tb = self.time[-1]

    #     self.loaded_thrust = True

    
    # def __repr__(self):

    #     return str([self.thrust, self.time])
    

    # def calc_thrust(self, time):
    #     thrust = np.interp(time, self.time, self.thrust, right=0)
    #     return thrust
    
    # def plot_thrust(self, units = 'N'):

            
    #     t = np.linspace(0, self.tb, 1000)

    #     T = np.array([self.calc_thrust(i) for i in t])

    #     if units == 'lbf':
    #         T *= N2LBF
        
    #     plt.xlabel(f'Time (s)')
    #     plt.ylabel(f'Thrust ({units})')
    #     plt.xlim(-self.tb * 0.1, self.tb * 1.1)
    #     plt.ylim(0, max(T) * 1.1)
    #     plt.title(f"Thrust of '{self.name}'")
    #     plt.plot(t,T)
    #     plt.show()

    #     # self.thrust[self.step] = interpolate.interp1d(self.motor['seconds'], self.motor['thrust'], bounds_error = False, fill_value=0)(t) # N

    


