'''
This file illustrates the use of the waterheaters library to run a test case

It loads the main function from the library and provides as parameters the input excel file and the folder where
the results should be written
'''
import os,sys
# include the main library path (the parent folder) in the path environment variable
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)

# import the library as a package (defined in __init__.py) => function calls are done through the lpackage (eg om.solve_model)
import waterheaters as wh

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:48:14 2023

@author: nicol
"""

# import processFunctions as procF
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from scipy.interpolate import UnivariateSpline

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
    
For one sequence (constant mass flow, temperature setpoint), the outputs are the following (inside the class):
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
Diameter = 0.419 # m 
Height = 1.09 # m
double = False # Velis has two storage tank

# Model creation
NUOS = wh.procF.WaterHeater(Height = Height, Diameter = Diameter, EWH = False, HPWH = True, double = double)

# Definition of the heating system geometry
NUOS.Heating_system(z_control = 0.29, z_init_E = 0.0, z_init_HP = 0.05, height_E = 0.16, height_HP = 0.25, Q_dot_peak_E = 1000) # !! For one single tank

# Tank model empirical parameters 
NUOS.Model_parameters(h_amb = 0.7, h_ref = 430, delta = 10000, H_mix = 0.2, eps_is = 0.65, W_dot_el_basis = 100)

#MN model initialization
nx = 40

# T_init_exp_distrib = np.array([33.9976552530881,	36.24012980996889,	37.15446746879251,	
#                                38.71922175039015,	40.19951689043508,	40.271976265560816,
#                                41.1990615328445, 40.89044506455033,	41.34058547967568,	
#                                40.20550017251811,	40.58540274466435])
# T_init_exp_distrib = np.array([	48.06071048024077,	54.98944060936443,	54.184856946015366,
#                                54.207067362080096,	55.36653025919837,	54.4420522113191,
#                                55.02173622414427,	54.79476384317278,	55.13604682719688,
#                                53.665555949600694,	53.93961749601926])

# T_init_exp_distrib = np.flip(T_init_exp_distrib + 273.15)

# old_indices = np.arange(0,len(T_init_exp_distrib))
# new_indices = np.linspace(0,len(old_indices)-1,nx)
# spl = UnivariateSpline(old_indices,T_init_exp_distrib,k=3,s=0)
# T_vect_init = spl(new_indices)

T_init = 52.2 + 273.15
T_vect_init = np.full(nx, T_init)
# The first temperature initialization vector is used for the first tank (where inlet comes in) the second for the second tank
NUOS.MN_model_init(nx, T_vect_init)

# Simulation constant variables
T_w_supply = 15.39 + 273.15 # degC
T_amb = 15.99 + 273.15

### Input variable
T_SP = 0 + 273.15
T_ext = T_amb

#%% Simulation
plt.close("all")

dt_cha = 60 #time step of a charge
dt_dis = 2 #time step of a discharge
tic_model = time.perf_counter()

dt = dt_cha# seconds
time_sq = 70*3600 # seconds
m_dot = 0
NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10 minutes of discharge
# dt = dt_dis# seconds
# time_sq = 24316-23706 # seconds
# m_dot = 0.13107+0.131*0.025 # kg/s
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10000 seconds of rest
# dt = dt_cha# seconds
# time_sq = 72305-24316 # seconds
# m_dot = 0
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10 minutes of discharge
# dt = dt_dis# seconds
# time_sq = 72618-72305 # seconds
# m_dot = 0.13107+0.131*0.025 # kg/s
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10000 seconds of rest
# dt = dt_cha# seconds
# time_sq = 75905-72618 # seconds
# m_dot = 0
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10 minutes of discharge
# dt = dt_dis# seconds
# time_sq = 76517-75905 # seconds
# m_dot = 0.13107+0.131*0.025 # kg/s
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

# # 10000 seconds of rest
# dt = dt_cha# seconds
# time_sq = 24*3600 - 76517 # seconds
# m_dot = 0
# NUOS.MN_model(dt, time_sq, m_dot, T_w_supply, T_amb, T_SP, T_ext)

NUOS.plotConso()

toc_model = time.perf_counter()
time_sim = toc_model - tic_model

time_step = np.zeros(len(NUOS.time_vect) - 1)
for i in range(len(time_step)):
    time_step[i] = NUOS.time_vect[i+1] - NUOS.time_vect[i]

# VELIS.plotConso()
E_e_cons = sum(NUOS.W_dot_cons_tot*time_step)/3600/1000
E_amb_loss = -sum(NUOS.Q_dot_amb*time_step)/3600/1000
E_water_cons = sum(NUOS.Q_dot_water_used*time_step)/3600/1000

#%% Time taken
toc = time.perf_counter()
time_tot = toc - tic

print('Number of layers : ', nx)
print("E_e_cons", E_e_cons, ' kWh')
print("E_amb_loss", E_amb_loss, ' kWh')
print("E_water_cons", E_water_cons, ' kWh')
print('Simulation time:', str(time_tot), 's')

t_tot = toc - tic
len_building = len(NUOS.perf_building)
len_solving = len(NUOS.perf_solving)
t_avg_building = sum(NUOS.perf_building)/len(NUOS.perf_building)
t_avg_solving = sum(NUOS.perf_solving)/len(NUOS.perf_solving)
t_tot_building = sum(NUOS.perf_building)
t_tot_solving = sum(NUOS.perf_solving)
t_tot_sim = t_tot_building + t_tot_solving
t_remaining = time_tot - t_tot_sim
# print('Simulation time only:', str(t_tot_sim), 's')
print('Building time only:', str(t_tot_building), 's')
print('Solving time only:', str(t_tot_solving), 's')
print('Remaining time:', str(t_remaining), 's')

#%% SAVE RESULTS 

name = 'Ambient_losses_V2'
   
wh.procF.save_results(name, NUOS)
    