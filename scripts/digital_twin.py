# #%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
import os,sys
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import time
# Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_hour': 9, 'end_hour': 10, 'power_to_cut': 500},  # Reducir 50 kW entre 9:00 y 10:00
#     # {'start_hour': 12, 'end_hour': 13, 'power_to_cut': 30},  # Reducir 30 kW entre 12:00 y 13:00
#     # {'start_hour': 18, 'end_hour': 20, 'power_to_cut': 70}   # Reducir 70 kW entre 18:00 y 20:00
# ]
# def get_sorted_temperatures_at_time(pool, t):
#     """
#     Get and sort the temperatures of the water heaters from highest to lowest at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
    
#     Returns
#     -------
#     sorted_temperatures : list
#         List of tuples containing the heater and its temperature, sorted by temperature in descending order.
#     """
#     # Combine heaters with their corresponding temperatures at time t
#     heaters_with_temp = list(zip(pool.pool_WH, [temp[t] for temp in pool.T_probe_2Dlist]))  # Use the temperature at time t
    
#     # Sort heaters by temperature in descending order
#     sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1][0], reverse=True)  # Sort by the first temperature in the list
    
#     return sorted_heaters
# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP = N_random_HP, N_random_E = N_random_E, N_VELIS = N_VELIS, N_NUOS = N_NUOS,
#                                  nx = nx, T_w_supply = T_w_supply, T_amb = T_amb, control_strategy='full_load')
# pool.generate_pool() 
# # pool.simulate_pool_parallel(NDay) # Can be used when using the embedded control function

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55+273.15 , 55+273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)
# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0 #I Reset the variable cumulating the power of each WH7
#     # print(pool.T_probe_2Dlist)

    
#     for cnt_wh, WH in enumerate(pool.pool_WH):
        
#         # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#         # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#         # The second temperatrure in the tuple T_probe is the temperature of the second tank of the Velis
        
#         T_SP = 55 + 273.15
#         # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#         # switch1, switch2 = pool.control_functions(WH, t*60, T_probe[cnt_wh], T_SP, strategy = pool.pool_control_strategy)
#         switch1, switch2 = pool.control_functions(WH, t*60, T_probe[cnt_wh], T_SP, strategy = pool.pool_control_strategy)
        
#         # Simulate the water cnt_wh th water heater of the pool
#         T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#         sorted_heaters = get_sorted_temperatures_at_time(pool, t)
#         print(f"Time step {t}:")
#         for heater, temp in sorted_heaters:
#             print(f"  Temperature: {temp[0]}")
#     pool.record_results(t)
# print(f'Ratio of temperature constraints ({pool.T_constraint -273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')        
        
# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# # sorted_heaters = get_sorted_temperatures(pool)
# # for heater, temp in sorted_heaters:
# #     print(f"Temperature: {temp[0]}")

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')

#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
# import os, sys
# root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_folder)
# import time
# # Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
# import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute':10*60, 'end_minute': 12*60, 'heaters_to_turn_off': 0},  # Apagar 1 calentador entre las 8:00 y 9:00
#     # {'start_minute': 12*60, 'end_minute': 13*60, 'heaters_to_turn_off': 2},  # Apagar 2 calentadores entre las 12:00 y 13:00
#     # {'start_minute': 18*60, 'end_minute': 20*60, 'heaters_to_turn_off': 3}   # Apagar 3 calentadores entre las 18:00 y 20:00
# ]



# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()
# pool.simulate_pool_parallel(NDay)  # Simulate the pool in parallel


# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55+273.15 , 55+273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # Define the function to get and sort temperatures
# def get_sorted_temperatures_at_time(pool, t):
#     """
#     Get and sort the temperatures of the water heaters from highest to lowest at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
    
#     Returns
#     -------
#     sorted_temperatures : list
#         List of tuples containing the heater and its temperature, sorted by temperature in descending order.
#     """
#     # Combine heaters with their corresponding temperatures at time t
#     heaters_with_temp = list(zip(pool.pool_WH, [temp[t] for temp in pool.T_probe_2Dlist]))  # Use the temperature at time t
    
#     # Sort heaters by temperature in descending order
#     sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1][0], reverse=True)  # Sort by the first temperature in the list
    
#     return sorted_heaters

# # Define the function to turn off heaters by temperature
# def turn_off_heaters_by_temperature(pool, t, heaters_to_turn_off):
#     """
#     Turn off a specified number of heaters based on their temperature at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
#     heaters_to_turn_off : int
#         The number of heaters to turn off.
    
#     Returns
#     -------
#     None.
#     """
#     # Get and sort the temperatures of the heaters at time t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
    
#     # Turn off the specified number of heaters with the highest temperatures
#     for i in range(heaters_to_turn_off):
#         heater, temp = sorted_heaters[i]
#         heater.switch1 = False  # Turn off the heater
#         print(f"Turned off heater with temperature {temp[0]} at time step {t}")
# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7

#     # Check if there is a demand reduction at the current time
#     current_minute = t  # Use time in minutes directly
#     for schedule in demand_reduction_schedule:
#         if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#             turn_off_heaters_by_temperature(pool, t, schedule['heaters_to_turn_off'])

#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#         # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#         # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#         T_SP = 55 + 273.15
#         # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#         switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#         # Simulate the water cnt_wh th water heater of the pool
#         T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#     pool.record_results(t)
    
#     # Obtener y ordenar las temperaturas de los calentadores en el tiempo t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
#     # print(f"Time step {t}:")
#     # for heater, temp in sorted_heaters:
#     #     print(f"  Temperature: {temp[0]}")  # Imprimir la primera temperatura en la lista

# print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')

# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
# import os, sys
# root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_folder)
# import time
# # Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
# import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute': 8*60, 'end_minute': 9*60, 'heaters_to_turn_off': 10},  # Apagar 1 calentador entre las 8:00 y 9:00
#     # {'start_minute': 12*60, 'end_minute': 13*60, 'heaters_to_turn_off': 2},  # Apagar 2 calentadores entre las 12:00 y 13:00
#     # {'start_minute': 18*60, 'end_minute': 20*60, 'heaters_to_turn_off': 3}   # Apagar 3 calentadores entre las 18:00 y 20:00
# ]

# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()
# pool.simulate_pool_parallel(NDay)  # Simulate the pool in parallel

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55+273.15 , 55+273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # Define the function to get and sort temperatures
# def get_sorted_temperatures_at_time(pool, t):
#     """
#     Get and sort the temperatures of the water heaters from highest to lowest at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
    
#     Returns
#     -------
#     sorted_temperatures : list
#         List of tuples containing the heater and its temperature, sorted by temperature in descending order.
#     """
#     # Combine heaters with their corresponding temperatures at time t
#     heaters_with_temp = list(zip(pool.pool_WH, [temp[t] for temp in pool.T_probe_2Dlist]))  # Use the temperature at time t
    
#     # Sort heaters by temperature in descending order
#     sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1][0], reverse=True)  # Sort by the first temperature in the list
    
#     return sorted_heaters

# # Define the function to turn off heaters by temperature
# def turn_off_heaters_by_temperature(pool, t, heaters_to_turn_off):
#     """
#     Turn off a specified number of heaters based on their temperature at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
#     heaters_to_turn_off : int
#         The number of heaters to turn off.
    
#     Returns
#     -------
#     None.
#     """
#     # Get and sort the temperatures of the heaters at time t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
    
#     # Turn off the specified number of heaters with the highest temperatures
#     for i in range(heaters_to_turn_off):
#         heater, temp = sorted_heaters[i]
#         heater.switch1 = True  # Turn off the heater
#         print(f"Turned off heater with temperature {temp[0]} at time step {t}")

# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7

#     # Check if there is a demand reduction at the current time
#     current_minute = t  # Use time in minutes directly
#     for schedule in demand_reduction_schedule:
#         if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#             turn_off_heaters_by_temperature(pool, t, schedule['heaters_to_turn_off'])

#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#         # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#         # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#         T_SP = 55 + 273.15
#         # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#         switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#         # Simulate the water cnt_wh th water heater of the pool
#         T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#     pool.record_results(t)
    
#     # Obtener y ordenar las temperaturas de los calentadores en el tiempo t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
#     # print(f"Time step {t}:")
#     # for heater, temp in sorted_heaters:
#     #     print(f"  Temperature: {temp[0]}")  # Imprimir la primera temperatura en la lista

# print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')

# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')









#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
# import os, sys
# import pandas as pd
# root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_folder)
# import time
# import matplotlib.pyplot as plt
# # Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
# import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute': 8*60, 'end_minute': 10*60, 'heaters_to_turn_off': 10},  # Apagar 1 calentador entre las 8:00 y 9:00
#     # {'start_minute': 12*60, 'end_minute': 13*60, 'heaters_to_turn_off': 2},  # Apagar 2 calentadores entre las 12:00 y 13:00
#     # {'start_minute': 18*60, 'end_minute': 20*60, 'heaters_to_turn_off': 3}   # Apagar 3 calentadores entre las 18:00 y 20:00
# ]

# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # DataFrames to store temperatures and turned off heaters
# df_temperatures = pd.DataFrame(columns=['time', 'sorted_heaters'])
# df_turned_off = pd.DataFrame(columns=['time', 'turned_off_heaters'])

# # Define the function to get and sort temperatures
# def get_sorted_temperatures_at_time(pool, t):
#     """
#     Get and sort the temperatures of the water heaters from highest to lowest at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
    
#     Returns
#     -------
#     sorted_temperatures : list
#         List of tuples containing the heater and its temperature, sorted by temperature in descending order.
#     """
#     # Combine heaters with their corresponding temperatures at time t
#     heaters_with_temp = list(zip(pool.pool_WH, [temp[t] for temp in pool.T_probe_2Dlist]))  # Use the temperature at time t
    
#     # Sort heaters by temperature in descending order
#     sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1][0], reverse=True)  # Sort by the first temperature in the list
    
#     return sorted_heaters

# # Define the function to turn off heaters by temperature
# def turn_off_heaters_by_temperature(pool, t, heaters_to_turn_off):
#     """
#     Turn off a specified number of heaters based on their temperature at a specific time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
#     t : int
#         The time step to get the temperatures for.
#     heaters_to_turn_off : int
#         The number of heaters to turn off.
    
#     Returns
#     -------
#     turned_off_heaters : list
#         List of heaters that were turned off.
#     """
#     # Get and sort the temperatures of the heaters at time t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
    
#     turned_off_heaters = []
#     # Turn off the specified number of heaters with the highest temperatures
#     for i in range(heaters_to_turn_off):
#         heater, temp = sorted_heaters[i]
#         heater.switch1 = True  # Turn off the heater
#         turned_off_heaters.append((heater, temp[0]))
#         print(f"Turned off heater with temperature {temp[0]} at time step {t}")
    
#     return turned_off_heaters

# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7

#     # Check if there is a demand reduction at the current time
#     current_minute = t  # Use time in minutes directly
#     turned_off_heaters = []
#     for schedule in demand_reduction_schedule:
#         if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#             turned_off_heaters = turn_off_heaters_by_temperature(pool, t, schedule['heaters_to_turn_off'])

#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#         # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#         # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#         T_SP = 55 + 273.15
#         # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#         # switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#         # Simulate the water cnt_wh th water heater of the pool
#         T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#     pool.record_results(t)
    
#     # Obtener y ordenar las temperaturas de los calentadores en el tiempo t
#     sorted_heaters = get_sorted_temperatures_at_time(pool, t)
#     print(f"Time step {t}:")
#     for heater, temp in sorted_heaters:
#         print(f"  Temperature: {temp[0]}")  # Imprimir la primera temperatura en la lista
    
#     # Store temperatures in DataFrame
#     df_temperatures = pd.concat([df_temperatures, pd.DataFrame({'time': [t], 'sorted_heaters': [sorted_heaters]})], ignore_index=True)
    
#     # Store turned off heaters in DataFrame
#     if turned_off_heaters:
#         df_turned_off = pd.concat([df_turned_off, pd.DataFrame({'time': [t], 'turned_off_heaters': [turned_off_heaters]})], ignore_index=True)

# print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')

# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# # Plot temperatures of each heater
# def plot_temperatures(pool):
#     """
#     Plot the temperatures of each water heater over time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
    
#     Returns
#     -------
#     None.
#     """
#     time_vect = pool.time_vect_com
#     for i, temp_list in enumerate(pool.T_probe_2Dlist):
#         temperatures = [temp[0] for temp in temp_list]  # Extract the first temperature from each tuple
#         plt.plot(time_vect, temperatures, label=f'Heater {i+1}')
    
#     plt.xlabel('Time (s)')
#     plt.ylabel('Temperature (K)')
#     plt.title('Temperatures of Each Water Heater Over Time')
#     plt.legend()
#     plt.show()

# plot_temperatures(pool)

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')

# # Save DataFrames to CSV
# df_temperatures.to_csv('sorted_temperatures.csv', index=False)
# df_turned_off.to_csv('turned_off_heaters.csv', index=False)


#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable

####### Working #####

# import os, sys
# import pandas as pd
# root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_folder)
# import time
# import matplotlib.pyplot as plt
# # Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
# import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute': 8*60, 'end_minute': 13*60, 'heaters_to_turn_off': 10},  # Apagar 10 calentadores entre las 8:00 y 24:00
# ]

# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # DataFrames to store temperatures and turned off heaters
# df_temperatures = pd.DataFrame(columns=['time', 'heater', 'temperature'])
# df_turned_off = pd.DataFrame(columns=['time', 'heater'])

# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7

#     # Check if there is a demand reduction at the current time
#     current_minute = t  # Use time in minutes directly
#     turned_off_heaters = []
#     for schedule in demand_reduction_schedule:
#         if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#             # Turn off the specified number of heaters
#             for i in range(schedule['heaters_to_turn_off']):
#                 heater = pool.pool_WH[i]
#                 heater.switch1 = False  # Turn off the heater
#                 turned_off_heaters.append(heater)
#                 print(f"Turned off heater {i+1} at time step {t}")

#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         # Implement control strategy only if the heater is not turned off
#         if WH not in turned_off_heaters:
#             # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#             # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#             # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#             T_SP = 55 + 273.15
#             # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#             switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#             # Simulate the water cnt_wh th water heater of the pool
#             T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#         else:
#             # Ensure the heater remains off
#             T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, False, False)
#     pool.record_results(t)
    
#     # Store temperatures in DataFrame
#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         df_temperatures = pd.concat([df_temperatures, pd.DataFrame({'time': [t], 'heater': [cnt_wh], 'temperature': [T_probe[cnt_wh][0]]})], ignore_index=True)
    
#     # Store turned off heaters in DataFrame
#     if turned_off_heaters:
#         for heater in turned_off_heaters:
#             df_turned_off = pd.concat([df_turned_off, pd.DataFrame({'time': [t], 'heater': [pool.pool_WH.index(heater)]})], ignore_index=True)

# print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')

# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# # Plot temperatures of each heater
# def plot_temperatures(pool):
#     """
#     Plot the temperatures of each water heater over time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
    
#     Returns
#     -------
#     None.
#     """
#     time_vect = pool.time_vect_com
#     for i, temp_list in enumerate(pool.T_probe_2Dlist):
#         temperatures = [temp[0] for temp in temp_list]  # Extract the first temperature from each tuple
#         plt.plot(time_vect, temperatures, label=f'Heater {i+1}')
    
#     plt.xlabel('Time (s)')
#     plt.ylabel('Temperature (K)')
#     plt.title('Temperatures of Each Water Heater Over Time')
#     plt.legend()
#     plt.show()

# plot_temperatures(pool)

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')

# # Save DataFrames to CSV
# df_temperatures.to_csv('sorted_temperatures.csv', index=False)
# df_turned_off.to_csv('turned_off_heaters.csv', index=False)



#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
import os, sys
import pandas as pd
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import time
import matplotlib.pyplot as plt
# Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import source as procF
import plotly.graph_objects as go

#%% Simulation
# Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 10
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute': 5*60, 'end_minute': 12*60, 'heaters_to_turn_off': 5},  # Apagar 10 calentadores entre las 8:00 y 24:00
# ]

# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # DataFrames to store temperatures and turned off heaters
# df_temperatures = pd.DataFrame(columns=['time', 'heater', 'temperature'])
# df_turned_off = pd.DataFrame(columns=['time', 'heater'])

# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     # Loop over the water heaters
#     pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7

#     # Check if there is a demand reduction at the current time
#     current_minute = t  # Use time in minutes directly
#     turned_off_heaters = []
#     for schedule in demand_reduction_schedule:
#         if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#             # Get and sort the temperatures of the heaters at time t
#             heaters_with_temp = list(zip(pool.pool_WH, [temp[t] for temp in pool.T_probe_2Dlist]))
#             print(heaters_with_temp)  # Use the temperature at time t
#             sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1][0], reverse=True)  # Sort by the first temperature in the list
#             # print(sorted_heaters)
#             # Turn off the specified number of heaters with the highest temperatures
#             for i in range(schedule['heaters_to_turn_off']):
#                 heater, temp = sorted_heaters[i]
#                 heater.switch1 = False  # Turn off the heater
#                 turned_off_heaters.append(heater)
#                 # print(f"Turned off heater with temperature {temp[0]} at time step {t}")

#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         # Implement control strategy only if the heater is not turned off
#         if WH not in turned_off_heaters:
#             # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#             # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#             # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#             T_SP = 55 + 273.15
#             # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#             switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#             # Simulate the water cnt_wh th water heater of the pool
#             T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#         else:
#             # Ensure the heater remains off
#             T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, False, False)
#     pool.record_results(t)
    
#     # Store temperatures in DataFrame
#     for cnt_wh, WH in enumerate(pool.pool_WH):
#         df_temperatures = pd.concat([df_temperatures, pd.DataFrame({'time': [t], 'heater': [cnt_wh], 'temperature': [T_probe[cnt_wh][0]]})], ignore_index=True)
    
#     # Store turned off heaters in DataFrame
#     if turned_off_heaters:
#         for heater in turned_off_heaters:
#             df_turned_off = pd.concat([df_turned_off, pd.DataFrame({'time': [t], 'heater': [pool.pool_WH.index(heater)]})], ignore_index=True)

# print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')

# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')

# # Plot temperatures of each heater
# def plot_temperatures(self):
#     """
#     Plot the temperatures of each water heater over time.
    
#     Returns
#     -------
#     None.
#     """
#     fig = go.Figure()
    
#     for i, temp_list in enumerate(self.T_probe_2Dlist):
#         temperatures = [temp[0] for temp in temp_list]  # Extract the first temperature from each tuple
#         fig.add_trace(go.Scatter(
#             x=self.time_vect_com/3600,
#             y=temperatures,
#             mode='lines',
#             name=f'Heater {i+1}',
#             line=dict(color=f'rgba({i*25 % 255}, {i*50 % 255}, {i*75 % 255}, 0.8)')
#         ))
    
#     fig.update_layout(
#         title='Temperatures of Each Water Heater Over Time',
#         xaxis_title='Time (s)',
#         yaxis_title='Temperature (K)',
#         hovermode='x unified'
#     )
    
#     fig.show()

# plot_temperatures(pool)

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')

# # Save DataFrames to CSV
# df_temperatures.to_csv('sorted_temperatures.csv', index=False)
# df_turned_off.to_csv('turned_off_heaters.csv', index=False)


#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable


#%% Simulation
# Time counter start
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable



# import os, sys
# import pandas as pd
# root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(root_folder)
# import time
# import matplotlib.pyplot as plt
# # Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
# import source as procF

# #%% Simulation
# # Time counter start
# tic = time.perf_counter()

# # Number of water heater and types
# N_random_HP = 0
# N_random_E = 100
# N_VELIS = 0
# N_NUOS = 0

# # Number of day simulated  
# NDay = 1

# # Add pool inputs
# nx = 40 # Number of cell in each water heater
# T_amb = 19 + 273.15
# T_w_supply = 14 + 273.15
# demand_reduction_schedule = [
#     {'start_minute': 8*60, 'end_minute': 11*60, 'heaters_to_turn_off': 80},  # Apagar x calentadores entre las 8:00 y 24:00
# ]

# # Creation of the pool of water heater
# pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
#                                  nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
# pool.generate_pool()

# # Simulate the pool
# switch1 = False
# switch2 = False
# T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)
# pool.initialize_sim(NDay)

# # DataFrames to store temperatures and turned off heaters
# df_temperatures = pd.DataFrame(columns=['time', 'heater', 'temperature'])
# df_turned_off = pd.DataFrame(columns=['time', 'heater'])
# data = []
# # Loop over the time 
# for t in range(len(pool.time_vect_com)):
#     pool.P_el_vect_cum = 0
#     turned_off_heaters = []
#     step_data = []
#     if t > 0:
#         for cnt_wh, WH in enumerate(pool.pool_WH):
#             temp1, temp2 = pool.T_probe_2Dlist[cnt_wh][t-1]
#             step_data.append({
#                 'heater_index': cnt_wh,
#                 'time_index': t,
#                 'temperature_1': temp1,
#                 'temperature_2': temp2,
#                 'heater': pool.pool_WH[cnt_wh]
#             })
       
#         # Sort the data for the current time step by temperature_1 in descending order
#         step_data_sorted = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)
#         # print(step_data_sorted)
#         data.extend(step_data_sorted) 
#         current_minute = t  # Use time in minutes directly
#         for schedule in demand_reduction_schedule:
#             if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#                 # Turn off the specified number of heaters with the highest temperatures
#                 for i in range(schedule['heaters_to_turn_off']):
#                     heater_info = step_data_sorted[i]
#                     heater = heater_info['heater']
#                     heater.switch1 = False  # Turn off the heater
#                     turned_off_heaters.append(heater)
#     for cnt_wh, WH in enumerate(pool.pool_WH):

#         # print(cnt_wh)
#         # print(WH)
#         # Implement control strategy only if the heater is not turned off
#             if WH not in turned_off_heaters:
#                 # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
#                 # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
#                 # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis

#                 T_SP = 55 + 273.15
#                 # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
#                 switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)

#                 # Simulate the water cnt_wh th water heater of the pool
#                 T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
#             else:
#                 # Ensure the heater remains off
#                 T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, False, False)
#     pool.record_results(t)    
#     # else:

#     # Loop over the water heaters
#         # if t > 0:
#             # print(pool.T_probe_2Dlist[cnt_wh][t-1])


#     # pool.P_el_vect_cum = 0  # Reset the variable cumulating the power of each WH7
#     # print(pool.T_probe_2Dlist)
#     # Check if there is a demand reduction at the current time
#     # current_minute = t  # Use time in minutes directly
    
#     #     turned_off_heaters = []
#     # for schedule in demand_reduction_schedule:
#     #     if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#     #         # Turn off the specified number of heaters
#     #         for i in range(schedule['heaters_to_turn_off']):
#     #             heater = pool.pool_WH[i]
#     #             heater.switch1 = False  # Turn off the heater
#     #             turned_off_heaters.append(heater)
#                 # print(f"Turned off heater {i+1} at time step {t}")
#     # step_data = []
#     # for cnt_wh in range(len(pool.T_probe_2Dlist)):
#     #     temp1, temp2 = pool.T_probe_2Dlist[cnt_wh][t]
#     #     step_data.append({
#     #         'heater_index': cnt_wh,
#     #         'time_index': t,
#     #         'temperature_1': temp1,
#     #         'temperature_2': temp2,
#     #         'heater': pool.pool_WH[cnt_wh]
#     #     })

#     # # # Sort the data for the current time step by temperature_1 in descending order
#     # step_data_sorted = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)
   
#     # # Check if there is a demand reduction at the current time
#     # current_minute = t  # Use time in minutes directly
#     # for schedule in demand_reduction_schedule:
#     #     if schedule['start_minute'] <= current_minute < schedule['end_minute']:
#     #         # Turn off the specified number of heaters with the highest temperatures
#     #         for i in range(schedule['heaters_to_turn_off']):
#     #             heater_info = step_data_sorted[i]
#     #             heater = heater_info['heater']
#     #             heater.switch1 = False  # Turn off the heater
#     #             turned_off_heaters.append(heater)


#             # print(step_data_sorted)
#             # # Get and sort the temperatures of the heaters at time t
#             # # print(pool.T_probe_2Dlist[i][t][0])
#             # heaters_with_temp = list(zip(pool.pool_WH, [pool.T_probe_2Dlist[i][t][0] for i in range(len(pool.pool_WH))]))  # Use the first temperature at time t
#             # # print(heaters_with_temp)
#             # sorted_heaters = sorted(heaters_with_temp, key=lambda x: x[1], reverse=True)  # Sort by the first temperature in the list
#             # # print(f"Time step {t}: Sorted heaters: {sorted_heaters}")
            
#             # # Turn off the specified number of heaters with the highest temperatures
#             # for i in range(schedule['heaters_to_turn_off']):
#             #     heater, temp = sorted_heaters[i]
#             #     heater.switch1 = False  # Turn off the heater
#             #     turned_off_heaters.append(heater)
#             #     # print(f"Turned off heater with temperature {temp} at time step {t}")
#     # print(pool.T_probe_2Dlist)



#     # for cnt_wh in range(len(pool.T_probe_2Dlist)):
#     #     temp1, temp2 = pool.T_probe_2Dlist[cnt_wh][t]
#     #     step_data.append({
#     #         'heater_index': cnt_wh,
#     #         'time_index': t,
#     #         'temperature_1': temp1,
#     #         'temperature_2': temp2,
#     #         'heater': pool.pool_WH[cnt_wh]
#     #     })

#     # # Sort the data for the current time step by temperature_1 in descending order
#     # step_data_sorted = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)
#     # print(step_data_sorted)
    
#     # Store temperatures in DataFrame
#     # for cnt_wh, WH in enumerate(pool.pool_WH):
#     #     df_temperatures = pd.concat([df_temperatures, pd.DataFrame({'time': [t], 'heater': [cnt_wh], 'temperature': [T_probe[cnt_wh][0]]})], ignore_index=True)
    
#     # # Store turned off heaters in DataFrame
#     # if turned_off_heaters:
#     #     for heater in turned_off_heaters:
#     #         df_turned_off = pd.concat([df_turned_off, pd.DataFrame({'time': [t], 'heater': [pool.pool_WH.index(heater)]})], ignore_index=True)

    
    
#     # Sort the data for the current time step by temperature_1 in descending order
#     # step_data_sorted = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)

#     # Sort the data for the current time step by temperature_1 in descending order
#     # step_data_sorted = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)
   
#     # Append the sorted data to the main data list
#     # data.extend(step_data_sorted)
#     # print(data)
# # print(f'Ratio of temperature constraints ({pool.T_constraint - 273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')
# # df = pd.DataFrame(data)
# # df.to_csv('T_probe_2Dlist.csv', index=False)
# # Plot de variables and save the results
# pool.plot_consumption()
# pool.plot_available_storage()
# pool.save_results_csv('TEST_digital_twin')
# df_sorted = pd.DataFrame(data)   
# df_sorted.to_csv('step_data_sorted.csv', index=False) 
# # Plot temperatures of each heater
# def plot_temperatures(pool):
#     """
#     Plot the temperatures of each water heater over time.
    
#     Parameters
#     ----------
#     pool : WaterHeaterPool
#         The pool of water heaters.
    
#     Returns
#     -------
#     None.
#     """
#     time_vect = pool.time_vect_com
#     for i, temp_list in enumerate(pool.T_probe_2Dlist):
#         temperatures = [temp[0] for temp in temp_list]  # Extract the first temperature from each tuple
#         plt.plot(time_vect/3600, temperatures, label=f'Heater {i+1}')
    
#     plt.xlabel('Time (s)')
#     plt.ylabel('Temperature (K)')
#     plt.title('Temperatures of Each Water Heater Over Time')
#     plt.legend()
#     plt.show()

# plot_temperatures(pool)

# toc = time.perf_counter()
# time_tot = toc - tic
# print('Simulation time:', str(time_tot), 's')

# # Save DataFrames to CSV
# df_temperatures.to_csv('sorted_temperatures.csv', index=False)
# df_turned_off.to_csv('turned_off_heaters.csv', index=False)



import os, sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import source as procF

# Initialize the simulation
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)

# Time counter start
tic = time.perf_counter()

# Number of water heater and types
N_random_HP = 0
N_random_E = 10
N_VELIS = 0
N_NUOS = 0

# Number of day simulated  
NDay = 1

# Add pool inputs
nx = 40  # Number of cell in each water heater
T_amb = 19 + 273.15
T_w_supply = 14 + 273.15
apagar = 1

demand_reduction_schedule = [
    {'start_minute': 7*60+30, 'end_minute':8*60+30, 'percentage_to_switch_off':1},  # Apagar 100% de la potencia disponible entre las 8:00 y 11:00
]
# switch_on_schedule = [
#     {'start_minute':12*60, 'end_minute': 16*60, 'power_to_switch_on': 1900},  
# ]
switch_on_schedule = [
    {'start_minute':3*60, 'end_minute': 5*60, 'power_to_switch_on': 0},  
]

csv_file_path = 'TEST_digital_twin_WH_charact_1.csv'
fixed_profiles_path = 'TEST_digital_twin_WH_timeseries_1.csv' 

# print(csv_file_path)
# Decide whether to load from CSV or generate a new pool
use_csv = True 
pool = procF.dlt.WaterHeaterPool(N_random_HP=N_random_HP, N_random_E=N_random_E, N_VELIS=N_VELIS, N_NUOS=N_NUOS,
                                 nx=nx, T_w_supply=T_w_supply, T_amb=T_amb, control_strategy='tracking_SP')
def read_csv_to_params(csv_file):
    df = pd.read_csv(csv_file, sep=';')
    params_list = []
    for _, row in df.iterrows():
       
        params = {
            'model': row['Type'],
            'volume': row['Volume (L)']/1000,
            'height': row['Height (m)'],
            'diameter': row['Diameter (m)'],
            'power': row['Electric Power (W)'],
            'EWH': True, 
            'HPWH': False,  
            'double': False,  
            'z_control': 0.3, 
            'z_init_E': 0.0,
            'z_init_HP': 0.0,
            'height_E': 0.3,
            'height_HP': 0.0,
            'Q_dot_peak_E': row['Electric Power (W)'],  # Usamos la potencia eléctrica como Q_dot_peak_E
            'h_amb': 0.75,  # Valores por defecto
            'H_mix': 0.15,
            'V_s': 0,
            'W_dot_el_basis': 0
        }
        params_list.append(params)
    return params_list


def create_pool_from_params(pool, params_list):
    pool.pool_WH = []
    for params in params_list:
        WH = pool.create_WH(params)
        pool.pool_WH.append(WH)

if use_csv and os.path.exists(csv_file_path):
    params_list = read_csv_to_params(csv_file_path)
    # print(params_list)
    create_pool_from_params(pool, params_list)
else:
    pool.generate_pool()
    # pool.save_pool_to_csv(csv_file_path)


# pool.simulate_pool_parallel(NDay, use_fixed_profiles=True, fixed_profiles_path=fixed_profiles_path)
# pool.load_fixed_profiles(fixed_profiles_path)
# Creation of the pool of water heater
# print(profiles)
# x= 4

# if use_csv and os.path.exists(csv_file_path):
#     pool.load_pool_from_csv(csv_file_path)
#     print(pool.load_pool_from_csv(csv_file_path))
# else:
#     pool.generate_pool()
#     pool.save_pool_to_csv(csv_file_path)

# pool.generate_pool()
# print(pool.)
# Simulate the pool
switch1 = False
switch2 = False
T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)
pool.initialize_sim(NDay)

# DataFrames to store temperatures and turned off heaters
df_temperatures = pd.DataFrame(columns=['time', 'heater', 'temperature'])
df_turned_off = pd.DataFrame(columns=['time', 'heater'])
data = []

# Available power list off is the power available to switch off and available power list on is the power available to switch on
available_power_list_off = []
available_power_list_on = []

# Loop over the time 
for t in range(len(pool.time_vect_com)):
    pool.P_el_vect_cum = 0
    turned_off_heaters = []
    turned_on_heaters = []
    step_data = []
    heaters_with_switch1_true = []
    if t > 0:
        for cnt_wh, WH in enumerate(pool.pool_WH):
            T_SP = 55 + 273.15
            switch1, switch2 = pool.control_functions(WH, t * 60, pool.T_probe_2Dlist[cnt_wh][t-1], T_SP, strategy=pool.pool_control_strategy)
            temp1, temp2 = pool.T_probe_2Dlist[cnt_wh][t-1]
            step_data.append({
                'heater_index': cnt_wh,
                'time_index': t,
                'temperature_1': temp1,
                'temperature_2': temp2,
                'heater': WH,
                'power':  int(switch1)*WH.param_heating["Q_dot_peak_E"]/1000,
                'power_on': int(not switch1)*WH.param_heating["Q_dot_peak_E"]/1000,
                'status': switch1
            })
            # if switch1:
            #     heaters_with_switch1_true.append(WH)
                # print(heaters_with_switch1_true)
        # pd.DataFrame(step_data).to_csv('step_data.csv', index=False)
        # heaters_with_switch1_true = [entry for entry in step_data if entry['status']]

        # Sort the data for the current time step by temperature_1 in descending order
        sort = True
        if sort == True:
            step_data_sorted_off = sorted(step_data, key=lambda x: x['temperature_1'], reverse=True)
            data.extend(step_data_sorted_off)
        else:
            step_data_sorted_off = step_data
            data.extend(step_data_sorted_off)
            # print(data)

        # Calculate the available power at time t
        available_power_off = sum([entry['power'] for entry in step_data_sorted_off])
        
        step_data_sorted_on = sorted(step_data, key=lambda x: x['temperature_1'])
        # available_power_list_off.append(available_power_off)
        #### Calculate the available power at time t to switch on
        available_power_on = sum([entry['power_on'] for entry in step_data_sorted_on])
        # print(available_power_on)
        # available_power_list_on.append(available_power_on)
        # print(available_power)
        # print(available_power)

        # Check each schedule to determine if the current time falls within the specified period
        current_minute = t  # Use time in minutes directly

        for schedule in demand_reduction_schedule:
            if schedule['start_minute'] <= current_minute < schedule['end_minute']:
                # print(schedule['start_minute'], current_minute, schedule['end_minute'])
                power_to_switch_off = available_power_off * schedule['percentage_to_switch_off']
                 
                power_switched_off = 0

                # Turn off heaters until the cumulative power to switch off is reached
                for entry in step_data_sorted_off:
                    # print(entry['heater'])
                    if power_switched_off >= power_to_switch_off:
                        break
                    
                    if entry['power'] <= (power_to_switch_off - power_switched_off):
                        # print(power_to_switch_off - power_switched_off)
                        # print(entry['power'])
                        heater = entry['heater']
                        heater.switch1 = False  # Turn off the heater
                        turned_off_heaters.append(heater)
                        power_switched_off += entry['power']
                        # print(power_switched_off)
                    else:
                        continue 
        # for schedule in switch_on_schedule:
        #     if schedule['start_minute'] <= current_minute < schedule['end_minute']:
        #             power_to_switch_on = available_power_on*schedule['percentage_to_switch_on']
        #             power_switch_on = 0
        #             for entry in step_data_sorted_on:
        #                 if power_switch_on >= power_to_switch_on:
        #                 # if power_switch_on >= power_to_switch_on:
        #                     break
        #                 heater = entry['heater']
        #                 # if entry['power_on'] <= (power_to_switch_on - power_switch_on):
        #                 heater.switch1 = True
        #                 turned_on_heaters.append(heater)
        #                 power_switch_on += entry['power_on']

        # for schedule in switch_on_schedule:
        #     if schedule['start_minute'] <= current_minute < schedule['end_minute']:
        #             power_to_switch_on = 1400#available_power_on*schedule['percentage_to_switch_on']
        #             power_switch_on = 0
        #             for entry in step_data_sorted_on:
        #                 if power_switch_on >= power_to_switch_on:
                    
        #                     break
        #                 heater = entry['heater']
        #                 heater.switch1 = True
        #                 turned_on_heaters.append(heater)
        #                 power_switch_on += entry['power_on']

        # for schedule in switch_on_schedule:
        #     if schedule['start_minute'] <= current_minute < schedule['end_minute']:
        #         power_to_switch_on = 1400  # available_power_on * schedule['percentage_to_switch_on']
        #         power_switch_on = 0
        #         for entry in step_data_sorted_on:
        #             if power_switch_on >= power_to_switch_on:
        #                 break
        #             heater = entry['heater']
        #             if heater in turned_on_heaters:  # Si el calentador ya está encendido, mantenerlo encendido
        #                 power_switch_on += entry['power_on']
        #             elif power_switch_on < power_to_switch_on:  # Si no está encendido y necesitamos más potencia
        #                 heater.switch1 = True
        #                 turned_on_heaters.append(heater)
        #                 power_switch_on += entry['power_on']

                #available_power_on*schedule['percentage_to_switch_on']
                
        # current_minute = t 
        for schedule in switch_on_schedule:
            if schedule['start_minute'] <= current_minute < schedule['end_minute']:
                    if schedule['power_to_switch_on']!=0:

                        heaters_with_switch1_true = [entry['heater'] for entry in step_data if entry['status']]
                        # print(heaters_with_switch1_true)
                        power_to_switch_on = schedule['power_to_switch_on']
                        power_switch_on = 0
                        for entry in step_data_sorted_on:
                        

                            if power_switch_on >= power_to_switch_on:                    
                                break
                            heater = entry['heater']
                            heater.switch1 = True
                            turned_on_heaters.append(heater)
                            power_switch_on += entry['power_on']


                    # for heater in heaters_with_switch1_true:
                    #     if heater not in turned_on_heaters:
                    #         heater.switch1 = True
                    #         turned_on_heaters.append(heater)
                    
                        # print(power_switch_on)                
                   

                            # print(len(heaters_with_switch1_true))





                        # else:
                        #     continue
    # print(turned_on_heaters)
    # Implement control strategy for all heaters
    for cnt_wh, WH in enumerate(pool.pool_WH):
        
        if WH in heaters_with_switch1_true:
                # print('aqi')
                T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, True , False)
                # heaters_with_switch1_true.remove(WH)
        elif WH in turned_on_heaters:
            # print('Heater is on')
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1 , False)
            WH.switch1 = False
            turned_on_heaters.remove(WH)
        
            # print('Im here')
            # for heater in turned_on_heaters:
        # elif WH in heaters_with_switch1_true:
        #     print('Heater is on')
        #     # print('Heater is on')
        #     # WH.switch1 = True
        #     T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1 , False)
                        # turned_on_heaters.append(heater)

        elif WH in turned_off_heaters:
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1 , False)
        # elif WH in turned_on_heaters and T_probe[cnt_wh][0] <= 80 + 273.15:
        #     T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1 , False)
        
            # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
            # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
            # The second temperature in the tuple T_probe is the temperature of the second tank of the Velis
       
        else:
            T_SP = 55 + 273.15
        # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
            switch1, switch2 = pool.control_functions(WH, t * 60, T_probe[cnt_wh], T_SP, strategy=pool.pool_control_strategy)
            Temp = T_probe[cnt_wh][0]
            # Simulate the water cnt_wh th water heater of the pool
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
            # print('here')



        
            
        
        
        
        # else:
        #     # if WH in turned_on_heaters:

        #     # # for schedule in switch_on_schedule:
        #     #      WH.switch1 = True
        #     #     break
        #     T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1 , False)
            # print(f"Time step {t}: Heater {cnt_wh} turned off")

    pool.record_results(t)  # Ensure results are recorded

    # Store temperatures in DataFrame
    
pool.plot_consumption()
pool.plot_available_storage()
pool.save_results_csv('TEST_digital_twin')

# Create a DataFrame from the collected data
df_sorted = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df_sorted.to_csv('step_data_sorted.csv', index=False)

# Display the DataFrame (optional)
# print(df_sorted)

# Save DataFrames to CSV
df_temperatures.to_csv('sorted_temperatures.csv', index=False)
df_turned_off.to_csv('turned_off_heaters.csv', index=False)

# Plot the available power at each time step
# plt.plot(range(len(available_power_list)), available_power_list)
# plt.xlabel('Time step')
# plt.ylabel('Available Power (W)')
# plt.title('Available Power at Each Time Step')
# plt.show()