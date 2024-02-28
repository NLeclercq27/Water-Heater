# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:48:14 2023

@author: nicol
"""

import processFunctions as procF
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

"""
The model takes as first inputs the storage dimensions:
    - Volume
    - Diameter
    - Presence of an Electrical Water Heater (EWH)
    - Presence of a Heat Pump Water Heater (HPWH)
Then, for a HPWH, the working fluid is needed + for both cases the heated fluid is also required
    - storage fluid
    - refrigerant
    
The heating system is then defined:
For a EWH, the electrical resistance peak power consumption is required
    - Q_dot_peak_E
The heating system dimensions are then required:
    - z_init_E (starting height of the electrical resistance)
    - z_init_HP (starting height of the heat pump coil)
    - height_E (height of the electrical resistance)
    - height_Hp (height of the HP coil)
    
The model empirical parameters need then to be entered: 
    - h_amb: heat exchange with the ambiance
    - h_ref: heat exchange with the refrigerant
    - delta: parameters mused to model the inversion of layer scheme 
    - H_mix: height of mixing with the supply water at the bottom of the storage tank
    - eps_is: isentropic efficiency of the compressor 
    
Then, it is necessary to initiate the simulation, with the number of division of the storage and the initial temperature distribution vector

    - nx: number of division for the MN model
    - T_vect_init --> initial vector of size nx
    
Then, the fisrt part of the simulation can be run, one needs to provide the water supply temperature, the ambient temperature 
Then, it is necessary to give the simulation time and time step as well as the mass flow rate for a discharge, and the temperature set point
If a heat pump is used, the external temperature (that can be equal to the ambient temperature) is required 
   - T_w_supply: water supply temperature
    - T_amb: ambiant temperature
    - time: total time of one simu
    - tau: time step 
    - m_dot: mass flow rate
    - T_SP: temeprature set point 
    - T_ext: for the heat pump mode
    
For one sequence (constant mass flow, temperature setpoint), the output are the following (inside the class):
    - self.t_vect: a vector of output time
    - self.Q_dot_cons_vec: a vector electrical consumption of the WH (in W) with the corresponding time Q_dot_cons_vect
    - self.T_record[-1]: the temperature profile inside the water storage tank after the sequence
If you add a sequence, the value will automatically be added to the previous sequence
"""

#%% Defintion of the parameters 

# TIme counter start
tic = time.perf_counter()

### Tank known parameters
# Volume = 0.065 # m^3
Diameter = 0.21 # m 
Height = 0.93 # m
double = True # Velis has two storage tank
fluid = 'water' 
refrigerant = 'propane' # only required if HPWH is TRUE (heat pump mode)

# Model creation
VELIS = procF.WaterHeater(fluid, Height = Height, Diameter = Diameter, EWH = True, HPWH = False, refrigerant = refrigerant, double = double)

# Definition of the heating system geometry
VELIS.Heating_system(z_init_E = 0.0, z_init_HP = 0.0, height_E = 0.3, height_HP = 0.25, Q_dot_peak_E = 1355.88) # !! For one single tank

# Tank model empirical parameters 
VELIS.Model_parameters(h_amb = 0.80, h_ref = 700, delta = 10000, s = 1, H_mix = 0.15, eps_is = 0.6)

#MN model initialization
nx = 40
T_init = 0 + 273.15 
T_vect_init = np.full(nx, T_init)
# The first temperature initialization vector is used for the first tank (where inlet comes in) the second for the second tank
VELIS.MN_model_init(nx, T_vect_init, T_vect_init)

# Simulation constant variables
T_w_supply = 15.25 + 273.15 # degC
T_amb = 17.60 + 273.15

### Input variable
T_SP = 60 + 273.15
T_ext = 10 + 273.15 #

#%% Simulation
plt.close("all")

dt_cha = 60 #time step of a charge
dt_dis = 2 #time step of a discharge
tic_model = time.perf_counter()
for i in range(1): # One day, can be changed to simulate several times the same day
    
    # First simulation 6 hours of heating
    dt = dt_cha# seconds
    time_sq = 3600*6 # seconds
    m_dot = 0
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)
    
    # Second simulation 10 minutes of hot water use (42 Liters in total)
    dt = dt_dis
    time_sq = 600
    m_dot = 0.08 # kg/s
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

    
    # Third simulation 11 h + 50 min hours of heating
    dt = dt_cha# seconds
    time_sq = 3600*11 + 3600*50/60 # seconds
    m_dot = 0# kg/s
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)
 
    # Fourth simulation 10 minutes of hot water use (42 Liters in total)
    dt = dt_dis
    time_sq = 600
    m_dot = 0.08 # kg/s
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)
    VELIS.plotTPointsVelis()
    # Fifth simulation 1h + 50 min hours of heating
    dt = dt_cha# seconds
    time_sq = 3600*1 + 3600*50/60 # seconds
    m_dot = 0# kg/s    
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

    # Sixth simulation 2 min of discharge
    dt = dt_dis# seconds
    time_sq = 120# seconds
    m_dot = 0.055# kg/s  
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

    # Seventh simulation 3h + 58 minutes of heating
    dt = dt_cha# seconds
    time_sq = 3600*3 + 58/60*3600# seconds
    m_dot = 0# kg/s    
    VELIS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

    
VELIS.plotTPointsVelis()
VELIS.plotConso()

toc_model = time.perf_counter()
time_sim = toc_model - tic_model

time_step = np.zeros(len(VELIS.time_vect) - 1)
for i in range(len(time_step)):
    time_step[i] = VELIS.time_vect[i+1] - VELIS.time_vect[i]

# VELIS.plotConso()
E_e_cons = sum(VELIS.W_dot_cons_tot*time_step)/3600/1000
E_amb_loss = -sum(VELIS.Q_dot_amb*time_step)/3600/1000
E_water_cons = sum(VELIS.Q_dot_water_used*time_step)/3600/1000

#%% Time taken
toc = time.perf_counter()
time_tot = toc - tic

print('Number of layers : ', nx)
print("E_e_cons", E_e_cons, ' kWh')
print("E_amb_loss", E_amb_loss, ' kWh')
print("E_water_cons", E_water_cons, ' kWh')
print('Simulation time:', str(time_tot), 's')

t_tot = toc - tic
len_building = len(VELIS.perf_building)
len_solving = len(VELIS.perf_solving)
t_avg_building = sum(VELIS.perf_building)/len(VELIS.perf_building)
t_avg_solving = sum(VELIS.perf_solving)/len(VELIS.perf_solving)
t_tot_building = sum(VELIS.perf_building)
t_tot_solving = sum(VELIS.perf_solving)
t_tot_sim = t_tot_building + t_tot_solving
t_remaining = time_tot - t_tot_sim
# print('Simulation time only:', str(t_tot_sim), 's')
print('Building time only:', str(t_tot_building), 's')
print('Solving time only:', str(t_tot_solving), 's')
print('Remaining time:', str(t_remaining), 's')

#%% SAVE RESULTS 

name = 'TEST'
list_file = ['C:\\Users\\nicol\\OneDrive - Universite de Liege\\Doctorat\\UNLEASH\Code\\Water_storage\\Simulations_results\\' , name ,'.pkl']

# The complete file name is built up
filename = "".join(list_file)       
   
# 'file' points at the file in which the pickled object will be written
with open(filename, "wb") as file:
    # The dump() method of the pickle module in Python, converts a Python object hierarchy into a byte stream. 
    # This process is also called as serilaization.
    pickle.dump(VELIS, file)
    

 





