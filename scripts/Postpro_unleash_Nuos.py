# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:55:12 2024

@author: nicol
"""

import pandas as pd
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
from pandas import concat
from scipy.interpolate import interp1d
import os,sys
# include the main library path (the parent folder) in the path environment variable
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import waterheaters as wh

plt.close('all')

#%% Function to open the TDMS file

def TDMS_to_CSV(filename,save_csv):
    tdms_file = TdmsFile(filename)
    group = tdms_file['Data']
    data = concat([pd.DataFrame(group[name][:]) for name in list(group)], axis=1)
    data.columns = list(group)
    if save_csv:
        csv_name = os.path.splitext(filename)[0] + '.csv'
        data.to_csv(csv_name)
    return data

#%% Function to display the temperature profile

def T_profile(data, start_index, t): 
    data["Chrono"] = data["Time"] - data["Time"][0]
    data["Chrono"] = data['Chrono'].dt.total_seconds() 
    data["Chrono"] = data["Chrono"]  - start_index

    h_vect = np.linspace(10, 90,9)
    index = (data['Chrono'] - t).abs().idxmin()
    row = data.loc[index]
    T_vect_1 = np.array([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 



    ## Lost on 2 hours of rest from one tank 

    t = 0 # seconds
    index = (data['Chrono'] - t).abs().idxmin()
    row = data.loc[index]
 
    ## Lost on 2 hours of rest from one tank 
    
    ## plot
    fontsize = 16
    fig,ax = plt.subplots(figsize=(3.5,6),constrained_layout=True)   
    plt.rcParams.update({'font.size':16})
    params = {
              "text.usetex" : True,
              "font.family" : "cm"}
    plt.rcParams.update(params)
    plt.grid()
    plt.xlim(10,90)
    plt.xticks((10,20,30,40,50,60,70,80,90))
    # plt.xticks(rotation=25)
    plt.xlabel('Temperature [$^\circ$C]', fontsize=fontsize)
    plt.ylabel('Height [cm]' ,fontsize=fontsize)      
    plt.plot(T_vect_1, h_vect,  c = u'#2ca02c' , linewidth = 2) 
    plt.scatter(T_vect_1, h_vect,  marker="o", s=26, color = u'#1f77b4')



#%% TEST

# """Read TDMS File"""
# data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\control_strategy.TDMS",0)
# data["Chrono"] = data["Time"] - data["Time"][0]
# data["Chrono"] = data['Chrono'].dt.total_seconds() 
# start_index = 0
# t= 7200

# # Plot the experimental temeprature profile at time t 
# T_profile(data, start_index, t)



# #%% Function to display experimental data varying with time 
# """Similar graph as in the code"""
# "Control strategy detection"

# #def plot_time_variation(data, t_start, t_stop):
# xdim = 5.5+1
# ydim = 4.3
# labelsize = 18

# time_vect = data["Chrono"]/3600


# ##### Test to check profile

# fontsize = 16
# fig,ax = plt.subplots(figsize=(9,2.5),constrained_layout=True)   
# plt.rcParams.update({'font.size':18})
# params = {
#           "text.usetex" : True,
#           "font.family" : "cm"}
# plt.rcParams.update(params)
# # plt.grid()
# # plt.xlim((0,24))
# # plt.xticks(rotation=25)
# plt.xlabel('Time [h]', fontsize=fontsize)
# plt.ylabel('Power [W]' ,fontsize=fontsize, color= u'#ff7f0e')   
# # plt.ylim(-10, 600)
# # plt.xlim(0, 25)
# # plt.yticks([0, 500,1000, 1500,2000])   
# l1 = ax.plot(time_vect, data['Power_Total'] , c = u'#ff7f0e' , linewidth = 2)   
# l2 = ax.plot(time_vect[::50], data['Power_Resistance'][::50], linestyle='dashed' , c = u'#ff7f0e' , linewidth = 2) 
# ax2=ax.twinx()   
# ax2.set_ylabel('Temperature [$^\circ$C]' ,fontsize=fontsize, color=  u'#2ca02c') 
# # ax2.set_ylim(70, 90) 
# # ax2.set_yticks([70,74,78,82,86,90])
# plt.grid()
# l3 = ax2.plot(time_vect,data['HP_PROBE'] , c = u'#2ca02c' , linewidth = 2) 
# l4 = ax2.plot(time_vect[::50],data['m_dot_w'][::50] , c = u'#2ca02c' , linestyle='dashed',linewidth = 2) 
# lns = l1 + l2 + l3 + l4
# labels = [ 'Total power','Power resistance','T control tank','Water flow rate']  
# # plt.legend(lns, labels,bbox_to_anchor=(0.985,1.02), ncol=4, fontsize=12)
# # labels = [ 'Oil Circulation Ratio','Pressure ratio']  
# # plt.legend(lns, labels,fontsize=14, loc='best')
# data['m_dot_avg'] = [ser['m_dot_w'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
# data['T_w_in_avg'] = [ser['HPT_IN'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
# m_dot_avg = np.mean(data['m_dot_avg'])
# T_w_in_avg = np.mean(data['T_w_in_avg'])
# T_amb_avg = np.mean(data['T_amb'])



#%% Heating system - time 

"""Read TDMS File"""

data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\Heating_to50_7hours_hp_only_lesspower.TDMS",0)
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 

T_amb_avg = np.mean(data['T_amb'])
t_min = 90 # delay due to temeprature sensors
t_max = 26900
#' from 2500 seconds to 2550 --> change of working conditions
Dt = t_max - t_min
index_min = (data['Chrono'] - t_min).abs().idxmin()
index_max = (data['Chrono'] - t_max).abs().idxmin()
index_vect = np.arange(index_min, index_max,1)
T_mean_exp = np.zeros(len(index_vect))

W_dot_e = data['Power_Total'][index_vect]

T_distrib = []

i = 0
for k in index_vect:
    row = data.loc[k]
    T_mean_exp[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'] , row['T_tank1_10'], row['T_tank1_11']]) 
    T_distrib.append([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'] , row['T_tank1_10'], row['T_tank1_11']])
    i +=1


time_vect_meanT = (data["Chrono"][index_min:index_max] - t_min)/3600
    


name = 'Heating_phase_HP'
path = '..\\data\\Simulations\\NUOS_valid\\'

results = wh.procF.open_results(name, path = path)

T_mean_sim = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    T_mean_sim[k] = np.mean(results.T_record1[k]) -273.15


# Avg temperature increase
fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)

plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

plt.plot(np.array(results.time_vect)/3600, T_mean_sim , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
plt.scatter(time_vect_meanT[::2000], T_mean_exp[::2000], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

plt.ylim([10,50])
plt.yticks([10,20,30,40,50])  
plt.xticks([0, 2,4,6,8])   
plt.legend(bbox_to_anchor=(0.6,1.02), ncol=1, fontsize=12)

# Error 
t_sim = np.array(results.time_vect)
T_sim = T_mean_sim
t_exp = time_vect_meanT*3600
T_exp = T_mean_exp

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, T_sim, kind='linear', fill_value='extrapolate')
T_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(T_sim_interpolated - T_exp))
print(f"Mean Absolute Error heating: {mae}")




# Power consumption
del results.time_vect[-1]  
fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)

plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Power consumtion [W]',fontsize=16)  

plt.plot(np.array(results.time_vect)/3600, results.W_dot_cons_tot , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
plt.scatter(time_vect_meanT[::500], W_dot_e[::500], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

 
plt.ylim([150,350])
plt.yticks([150,200,250,300,350])   
plt.xticks([0, 2,4,6,8])  
plt.legend(bbox_to_anchor=(0.6,1.02), ncol=1, fontsize=12)

# Error 
t_sim = np.array(results.time_vect)
W_dot_sim = results.W_dot_cons_tot
t_exp = time_vect_meanT*3600
W_dot_exp = W_dot_e

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, W_dot_sim, kind='linear', fill_value='extrapolate')
W_dot_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(W_dot_sim_interpolated - W_dot_exp))
print(f"Mean Absolute Error heating power [W]: {mae}")


#%% Ambiant losses 
"""Read TDMS File"""
data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\control_strategy.TDMS",0)
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 

T_amb_avg = np.mean(data['T_amb'])
t_min = 20200
t_max = 20200+70*3600
index_min = (data['Chrono'] - t_min).abs().idxmin()
index_max = (data['Chrono'] - t_max).abs().idxmin()
index_vect = np.arange(index_min, index_max,1)
T_mean_exp = np.zeros(len(index_vect))


i = 0
for k in index_vect:
    row = data.loc[k]
    T_mean_exp[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'], row['T_tank1_10'], row['T_tank1_11']]) 
    i +=1


time_vect_meanT = (data["Chrono"][index_min:index_max] - t_min)/3600
    


name = 'Ambient_losses_V2'

results = wh.procF.open_results(name, path = path)
V = results.Total_volume
E_min = results.E_min
E_max = results.E_max

E_sim = abs(results.E_power - results.E_power[0])

T_mean_sim = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    
    T_mean_sim[k] = np.mean(results.T_record1[k]) -273.15

fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)

plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

plt.plot(np.array(results.time_vect)/3600, T_mean_sim , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
plt.scatter(time_vect_meanT[::20000], T_mean_exp[::20000], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

plt.xlim([0,70])
plt.xticks([0, 10,20,30,40,50,60,70])   
plt.legend(bbox_to_anchor=(0.35,1.02), ncol=1, fontsize=12)
plt.ylim([35,55])
plt.yticks([35,40,45,50,55])   
# Error 
t_sim = np.array(results.time_vect)
T_sim = T_mean_sim
t_exp = time_vect_meanT*3600
T_exp = T_mean_exp

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, T_sim, kind='linear', fill_value='extrapolate')
T_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(T_sim_interpolated - T_exp))
print(f"Mean Absolute Error heat losses: {mae}")

#%% Time variation with flow rate -- temperature profile 
# """Read TDMS File"""

data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\water_discharge.TDMS",0)


start_index = 43
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 
data["Chrono"] = data["Chrono"]  - start_index
data['m_dot_avg'] = [ser['m_dot_w'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
data['T_w_in_avg'] = [ser['HPT_IN'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
m_dot_avg = np.mean(data['m_dot_avg'])
T_w_in_avg = np.mean(data['T_w_in_avg'])

name = 'water_discharge'
path = '..\\data\\Simulations\\NUOS_valid\\'
results = wh.procF.open_results(name, path = path)


t_minute = 12+180
t = t_minute*60
exp_data = {}
exp_data['h'] = np.linspace(10, 108,11)
index = (data['Chrono'] - t).abs().idxmin()
row = data.loc[index]
exp_data['T'] = np.array([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'], row['T_tank1_10'], row['T_tank1_11']]) 
exp_data['T_avg'] = np.mean(exp_data['T']) 
index_sim = np.argmin(abs(np.array(results.time_vect) - t))
results.plotTPointsNUOS(index = index_sim,exp_data = exp_data)


#%% 24h scenario


"""Read TDMS File"""
# data = TDMS_to_CSV("Experimental_results\\240212150139.TDMS",0)
data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\24H_testing.TDMS",0)
start_index = 0
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 
data["Chrono"] = data["Chrono"]  - start_index

data['m_dot_avg'] = [ser['m_dot_w'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
data['T_w_in_avg'] = [ser['HPT_IN'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
m_dot_avg = np.mean(data['m_dot_avg'])
T_w_in_avg = np.mean(data['T_w_in_avg'])
T_amb_avg = np.mean(data['T_amb'])

T_mean_exp = np.zeros(len(data["Chrono"]))
i = 0
for k in range(len(data["Chrono"])):
    row = data.loc[k]
    T_mean_exp[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'] , row['T_tank1_10'], row['T_tank1_11']]) 
    i +=1

"""Read simulation file"""
name = '24h_scenario'
path = '..\\data\\Simulations\\NUOS_valid\\'
results = wh.procF.open_results(name, path = path)
    
T_mean_sim = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    
    T_mean_sim[k] = np.mean(results.T_record1[k]) -273.15



W_dot_el = results.W_dot_cons_tot
  
    
#def plot_time_variation(data, t_start, t_stop):
xdim = 5.5+1
ydim = 4.3
labelsize = 18



time_vect = data["Chrono"]/3600
data['m_dot_w']  = [ser['m_dot_w'] +0.025*ser['m_dot_w'] if ser['m_dot_w'] > 10 else 0  for idx, ser in data.iterrows()]
data['T_w_in_avg'] = [ser['HPT_IN'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]

T_amb_avg = np.mean(data['T_amb'])
T_w_in_avg = np.mean(data['T_w_in_avg'])

##### Mass flow rate and avg temperature

fontsize = 16
fig,ax = plt.subplots(figsize=(5.5/1.4,4.3/1.5),constrained_layout=True)   
plt.rcParams.update({'font.size':16})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)
# plt.grid()
plt.xlim((0,24.1))
plt.xticks([0, 4,8,12,16,20,24])   
# plt.xticks(rotation=25)
plt.xlabel('Time [h]', fontsize=fontsize)
plt.ylabel('Avg temperature [$^\circ$C]' ,fontsize=fontsize, color= u'#ff7f0e')   
plt.ylim(20, 70)
# plt.xlim(0, 25)
plt.yticks([20,30, 40,50,60,70])   


ax.plot(np.array(results.time_vect)/3600, T_mean_sim, c = u'#ff7f0e' , linewidth = 2,zorder=1,label =  'Simulation') 
ax.scatter(time_vect[::2000], T_mean_exp[::2000], marker="o", s=36, color = 'r',zorder=2, label = 'Experimental')  
ax2=ax.twinx()   
ax2.set_ylabel('Mass flow rate [g/s]' ,fontsize=fontsize, color=  u'#1f77b4') 
ax2.set_ylim(0, 175) 
# ax2.set_yticks([70,74,78,82,86,90])
plt.grid()
ax2.plot(time_vect,data['m_dot_w'] , c = u'#1f77b4', linewidth = 2) 

ax.legend(bbox_to_anchor=(0.76,1.02), ncol=1, fontsize=12)

del results.time_vect[-1]  

##### Mass flow rate and power 

fontsize = 16
fig,ax = plt.subplots(figsize=(5.5/1.4,4.3/1.5),constrained_layout=True)   
plt.rcParams.update({'font.size':16})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)
# plt.grid()
plt.xlim((0,24.1))
plt.xticks([0, 4,8,12,16,20,24])   
# plt.xticks(rotation=25)
plt.xlabel('Time [h]', fontsize=fontsize)
plt.ylabel('Power [W]' ,fontsize=fontsize, color= u'#ff7f0e')   
plt.ylim(-40, 600)
# plt.xlim(0, 25)
plt.yticks([0, 100,200, 300,400,500,600])   


ax.plot(np.array(results.time_vect)/3600, W_dot_el , c = u'#ff7f0e' , linewidth = 2,zorder=1,label =  'Simulation') 
ax.scatter(time_vect[::2000], data['Power_Total'][::2000], marker="o", s=36, color = 'r',zorder=2, label = 'Experimental')  
ax2=ax.twinx()   
ax2.set_ylabel('Mass flow rate [g/s]' ,fontsize=fontsize, color=  u'#1f77b4') 
ax2.set_ylim(0, 175) 
# ax2.set_yticks([70,74,78,82,86,90])
plt.grid()
ax2.plot(time_vect,data['m_dot_w'] , c = u'#1f77b4', linewidth = 2) 

ax.legend(bbox_to_anchor=(0.76,1.02), ncol=1, fontsize=12)


# t = 3600*8
# exp_data = {}
# exp_data['h'] = np.linspace(10, 108,11)
# index = (data['Chrono'] - t).abs().idxmin()
# row = data.loc[index]
# exp_data['T'] = np.array([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'], row['T_tank1_10'], row['T_tank1_11']]) 
# exp_data['T_avg'] = np.mean(exp_data['T']) 
# index_sim = np.argmin(abs(np.array(results.time_vect) - t))
# results.plotTPointsNUOS(index = index_sim,exp_data = exp_data)


#%% heating scenari


"""Read TDMS File"""
# data = TDMS_to_CSV("Experimental_results\\240212150139.TDMS",0)
data_heating_25C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_25C.TDMS",0)
data_heating_20C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_20C.TDMS",0)
data_heating_15C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_15C.TDMS",0)
data_heating_10C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_10C.TDMS",0)
data_heating_5C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_5C.TDMS",0)
data_heating_0C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_0C.TDMS",0)
# data_heating_neg5C = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_HP\\heating_-5C.TDMS",0)

#%% Calculations

data_heating = {}
data_heating['25C'] = {}
data_heating['20C'] = {}
data_heating['15C'] = {}
data_heating['10C'] = {}
data_heating['5C'] = {}
data_heating['0C'] = {}
# data_heating['-5C'] = {}
data_heating['25C']['df'] = data_heating_25C
data_heating['20C']['df'] = data_heating_20C
data_heating['15C']['df'] = data_heating_15C
data_heating['10C']['df'] = data_heating_10C
data_heating['5C']['df'] = data_heating_5C
data_heating['0C']['df'] = data_heating_0C
# data_heating['-5C']['df'] = data_heating_neg5C

T_init = 15

for key in data_heating:
    data_heating[key]['df']["Chrono"] = data_heating[key]['df']["Time"] - data_heating[key]['df']["Time"][0]
    data_heating[key]['df']["Chrono"] = data_heating[key]['df']["Chrono"].dt.total_seconds() 
    
    data_heating[key]["T_amb_avg"] = np.mean(data_heating[key]['df']['T_amb'])
    data_heating[key]["W_dot_e"] = data_heating[key]['df']['Power_Total']
    
    # Avg temperature
    data_heating[key]['T_mean_exp'] = np.zeros(len(data_heating[key]['df']["Chrono"]))
    data_heating[key]['time'] = data_heating[key]['df']["Chrono"]
    E_tot = 0
    for k in range(len(data_heating[key]['df']["Chrono"])):
        row = data_heating[key]['df'].loc[k]
        data_heating[key]['T_mean_exp'][k] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9'] , row['T_tank1_10'], row['T_tank1_11']])
        E_tot += data_heating[key]['df']['Power_Total'][k]
    
    data_heating[key]['E_tot'] = E_tot/3.6/10**6
    
    diff = data_heating[key]['T_mean_exp'][0] - T_init
    data_heating[key]['T_mean_exp'] = data_heating[key]['T_mean_exp']  - diff
    
    
    
    
#%% Plots
fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.3),constrained_layout=True)   
plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)
plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

for key in data_heating:
    # plt.plot(np.array(results.time_vect)/3600, T_mean_sim , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
    # plt.scatter(time_vect_meanT[::2000], T_mean_exp[::2000], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  
    plt.plot(data_heating[key]['time'][::500]/3600, data_heating[key]['T_mean_exp'][::500],  linewidth = 1.5, label = key)  
plt.ylim([10,60])
plt.xlim([0,13])
plt.yticks([10,20,30,40,50,60])  
# plt.xticks([0, 2,4,6,8])   
# plt.legend(bbox_to_anchor=(0.6,1.02), ncol=6, fontsize=12)
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=14)


#####################################################################


fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.3),constrained_layout=True)   
plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)
plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Power consumtion [W]',fontsize=16)  
# plt.plot(np.array(results.time_vect)/3600, results.W_dot_cons_tot , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
# plt.scatter(time_vect_meanT[::500], W_dot_e[::500], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  
for key in data_heating:    
    plt.plot(data_heating[key]['time'][::500]/3600, data_heating[key]["W_dot_e"][::500] ,  linewidth = 1.5, label = key)  
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=14) 
plt.ylim([0,350])
plt.xlim([0,13])
# plt.yticks([])   
# plt.xticks([0, 2,4,6,8])  

# name = 'Heating_phase_HP'
# path = '..\\data\\Simulations\\NUOS_valid\\'

# results = wh.procF.open_results(name, path = path)

# T_mean_sim = np.zeros(len(results.T_record1))
# for k in range(len(results.T_record1)):
#     T_mean_sim[k] = np.mean(results.T_record1[k]) -273.15


# # Avg temperature increase
# fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

# plt.rcParams.update({'font.size':'16'})
# params = {
#           "text.usetex" : True,
#           "font.family" : "cm"}
# plt.rcParams.update(params)

# plt.grid()
# plt.xlabel('Time [h]', fontsize=16)
# plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

# plt.plot(np.array(results.time_vect)/3600, T_mean_sim , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
# plt.scatter(time_vect_meanT[::2000], T_mean_exp[::2000], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

# plt.ylim([10,50])
# plt.yticks([10,20,30,40,50])  
# plt.xticks([0, 2,4,6,8])   
# plt.legend(bbox_to_anchor=(0.6,1.02), ncol=1, fontsize=12)

# # Error 
# t_sim = np.array(results.time_vect)
# T_sim = T_mean_sim
# t_exp = time_vect_meanT*3600
# T_exp = T_mean_exp

# # Interpolate the simulated data over the experimental time points
# interpolator = interp1d(t_sim, T_sim, kind='linear', fill_value='extrapolate')
# T_sim_interpolated = interpolator(t_exp)

# # Calculate Mean Absolute Error (MAE)
# mae = np.mean(np.abs(T_sim_interpolated - T_exp))
# print(f"Mean Absolute Error heating: {mae}")




# # Power consumption
# del results.time_vect[-1]  
# fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

# plt.rcParams.update({'font.size':'16'})
# params = {
#           "text.usetex" : True,
#           "font.family" : "cm"}
# plt.rcParams.update(params)

# plt.grid()
# plt.xlabel('Time [h]', fontsize=16)
# plt.ylabel('Power consumtion [W]',fontsize=16)  

# plt.plot(np.array(results.time_vect)/3600, results.W_dot_cons_tot , color = 'k',  linewidth = 1.5,zorder=1,label =  'Simulation')  
# plt.scatter(time_vect_meanT[::500], W_dot_e[::500], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

 
# plt.ylim([150,350])
# plt.yticks([150,200,250,300,350])   
# plt.xticks([0, 2,4,6,8])  
# plt.legend(bbox_to_anchor=(0.6,1.02), ncol=1, fontsize=12)

