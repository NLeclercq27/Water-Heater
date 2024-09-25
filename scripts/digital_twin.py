#%% Aggregation of water heaters (digital twin) --> simulation script

# Imports
# Include the main library path (the parent folder) in the path environment variable
import os,sys
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import time
# Import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import source as procF

#%% Simulation
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
nx = 40 # Number of cell in each water heater
T_amb = 19 + 273.15
T_w_supply = 14 + 273.15


# Creation of the pool of water heater
pool = procF.dlt.WaterHeaterPool(N_random_HP = N_random_HP, N_random_E = N_random_E, N_VELIS = N_VELIS, N_NUOS = N_NUOS,
                                 nx = nx, T_w_supply = T_w_supply, T_amb = T_amb)
pool.generate_pool() 
# pool.simulate_pool_parallel(NDay) # Can be used when using the embedded control function

# Simulate the pool
switch1 = False
switch2 = False
T_probe = [(0.0 , 0.0)] * len(pool.pool_WH)
pool.initialize_sim(NDay)
# Loop over the time 
for t in range(len(pool.time_vect_com)):
    # Loop over the water heaters
    pool.P_el_vect_cum = 0 #I Reset the variable cumulating the power of each WH
    for cnt_wh, WH in enumerate(pool.pool_WH):
        
        # Implement control strategy any strategy can be used determining if switch 1 (heating resistor) 
        # and switch 2 (HP compressor) are activated or not based on the temperature of the probe in the tuple T_probe[cnt_wh]
        # The second temperatrure in the tuple T_probe is the temperature of the second tank of the Velis
        
        T_SP = 55 + 273.15
        # Default control strategy to track the setpoint with +3K -3K of hysteresis (see control_functions)
        switch1, switch2 = pool.control_functions(WH, t*60, T_probe[cnt_wh], T_SP, strategy = pool.pool_control_strategy)
        # Simulate the water cnt_wh th water heater of the pool
        T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)
    pool.record_results(t)
print(f'Ratio of temperature constraints ({pool.T_constraint -273.15:.2f}°C) respected: {int(sum(pool.T_constraint_bool_vect))}/{len(pool.pool_WH)}')        
        
# Plot de variables and save the results
pool.plot_consumption()
pool.plot_available_storage()
pool.save_results_csv('TEST_digital_twin')

toc = time.perf_counter()
time_tot = toc - tic
print('Simulation time:', str(time_tot), 's')











