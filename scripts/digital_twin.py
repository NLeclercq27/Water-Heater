#%% Aggregation of water heaters --> simulation

# imports
# include the main library path (the parent folder) in the path environment variable
import os,sys
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import time
# import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import source as procF

#%% Simulation
# Time counter start
tic = time.perf_counter()

# Number of water heater and types
N_random_HP = 1
N_random_E = 1
N_VELIS = 1
N_NUOS = 1

# Number of day simulated  
NDay = 1
 
# Creation of the pool of water heater
pool = procF.dlt.WaterHeaterPool(N_random_HP = N_random_HP, N_random_E = N_random_E, N_VELIS = N_VELIS, N_NUOS = N_NUOS) 
pool.generate_WH() 
pool.simulate_pool_parallel(NDay)
pool.plot_consumption()
pool.plot_available_storage()
pool.save_results_csv('TEST_digital_twin')


toc = time.perf_counter()
time_tot = toc - tic
print('Simulation time:', str(time_tot), 's')








                