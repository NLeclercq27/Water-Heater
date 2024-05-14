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
# from matplotlib import cm
import os,sys
# include the main library path (the parent folder) in the path environment variable
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import waterheaters as wh
sys.modules['processFunctions_bis'] = wh.procF
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
    T_vect_2 = np.array([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']])       


    ## Lost on 2 hours of rest from one tank 
    T_vect_2_means_7200 = np.mean(T_vect_2)
    t = 0 # seconds
    index = (data['Chrono'] - t).abs().idxmin()
    row = data.loc[index]
    T_vect_2 = np.array([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']])       
    ## Lost on 2 hours of rest from one tank 
    T_vect_2_means_0 = np.mean(T_vect_2)

    DT = T_vect_2_means_0 - T_vect_2_means_7200
    cp = 4187.1 #J/kg/K
    m_tot = 32.2*999.2887 # kg
    E_lost = m_tot*cp*DT
    
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
    plt.plot(T_vect_2, h_vect,  c = u'#2ca02c' , linewidth = 2) 
    plt.scatter(T_vect_2, h_vect,  marker="o", s=26, color = u'#1f77b4')





#%% Example

# """Read TDMS File"""
# data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results\\Control_strategy_def.TDMS",0)
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


# ##### power, temperature at sensor

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
# plt.ylim(-10, 2000)
# plt.xlim(0, 25)
# plt.yticks([0, 500,1000, 1500,2000])   
# l1 = ax.plot(time_vect, data['Power_el_res1'] , c = u'#ff7f0e' , linewidth = 2)   
# l2 = ax.plot(time_vect[::50], data['Power_el_res2'][::50], linestyle='dashed' , c = u'#ff7f0e' , linewidth = 2) 
# ax2=ax.twinx()   
# ax2.set_ylabel('Temperature [$^\circ$C]' ,fontsize=fontsize, color=  u'#2ca02c') 
# ax2.set_ylim(70, 90) 
# ax2.set_yticks([70,74,78,82,86,90])
# plt.grid()
# l3 = ax2.plot(time_vect,data['T_Tank1_PT100'] , c = u'#2ca02c' , linewidth = 2) 
# l4 = ax2.plot(time_vect[::50],data['T_Tank2_PT100'][::50] , c = u'#2ca02c' , linestyle='dashed',linewidth = 2) 
# lns = l1 + l2 + l3 + l4
# labels = [ 'Power tank 1','Power tank 2','T control tank 1','T control tank 2']  
# plt.legend(lns, labels,bbox_to_anchor=(0.985,1.02), ncol=4, fontsize=12)
# # labels = [ 'Oil Circulation Ratio','Pressure ratio']  
# # plt.legend(lns, labels,fontsize=14, loc='best')

#%% AVG temperature - time - heating system

"""Read TDMS File"""

data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_E\\24h_scenario.TDMS",0)
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 

T_amb_avg = np.mean(data['T_amb'])
t_min = 20
t_max = 14020
index_min = (data['Chrono'] - t_min).abs().idxmin()
index_max = (data['Chrono'] - t_max).abs().idxmin()
index_vect = np.arange(index_min, index_max,1)
T_mean_exp1 = np.zeros(len(index_vect))
T_mean_exp2 = np.zeros(len(index_vect))

i = 0
for k in index_vect:
    row = data.loc[k]
    T_mean_exp1[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 
    T_mean_exp2[i] = np.mean([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']]) 
    i +=1

T_mean_exp_80 = 0.5*T_mean_exp1 + 0.5*T_mean_exp2
time_vect_meanT = (data["Chrono"][index_min:index_max] - t_min)/3600
    


name = 'Heating_system'
path = '..\\data\\Simulations\\VELIS_valid\\'

results = wh.procF.open_results(name, path = path)
V = results.Total_volume
E_min = results.E_min
E_max = results.E_max

E_sim = abs(results.E_power - results.E_power[0])

T_mean_sim80 = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    
    T_mean_sim80[k] = 0.5*np.mean(results.T_record1[k]) + 0.5*np.mean(results.T_record2[k]) -273.15



fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)

plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

plt.plot(np.array(results.time_vect)/3600, T_mean_sim80 , color = 'k',  linewidth = 1.5,zorder=1)  
plt.scatter(time_vect_meanT[::1000], T_mean_exp_80[::1000], marker="o", s=36, color = 'r',zorder=2)  
plt.xlim(0)
plt.xticks((0,1,2,3,4))
plt.ylim(10,80)
plt.yticks((10,20,30,40,50,60,70,80))
# fmt = mdates.DateFormatter('%H:%M')
# ax.xaxis.set_major_formatter(fmt)

# Error 
t_sim = np.array(results.time_vect)
T_sim = T_mean_sim80
t_exp = time_vect_meanT*3600
T_exp = T_mean_exp_80

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, T_sim, kind='linear', fill_value='extrapolate')
T_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(T_sim_interpolated - T_exp))
print(f"Mean Absolute Error heating: {mae}")



# Power consumption
time_vect = data["Chrono"]/3600
del results.time_vect[-1]  


data['power_tot'] = data['Power_el_res1'] + data['Power_el_res2'] - 110


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
plt.scatter(time_vect_meanT[::500], data['power_tot'][index_min:index_max][::500], marker="o", s=36, color = 'r',zorder=2,label =  'Experimental')  

 
plt.ylim([1000,1500])
# plt.yticks([150,200,250,300,350])   
plt.xticks([0,1, 2,3])  
plt.xlim([0,3])  
plt.legend(bbox_to_anchor=(0.6,1.02), ncol=1, fontsize=12)


# Error 
t_sim = np.array(results.time_vect)
W_dot_sim = results.W_dot_cons_tot
t_exp = time_vect_meanT*3600
W_dot_exp = data['power_tot'][index_min:index_max]

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, W_dot_sim, kind='linear', fill_value='extrapolate')
W_dot_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(W_dot_sim_interpolated - W_dot_exp))
print(f"Mean Absolute Error heating power [W]: {mae}")


#%% AVG temperature - time - ambient losses

"""Read TDMS File"""
data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_E\\70h_ambiant_losses.TDMS",0)
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 


T_amb_avg = np.mean(data['T_amb'])
t_min = 0
t_max = 70*3600
index_min = (data['Chrono'] - t_min).abs().idxmin()
index_max = (data['Chrono'] - t_max).abs().idxmin()
index_vect = np.arange(index_min, index_max,1)
T_mean_exp1 = np.zeros(len(index_vect))
T_mean_exp2 = np.zeros(len(index_vect))

i = 0
for k in index_vect:
    row = data.loc[k]
    T_mean_exp1[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 
    T_mean_exp2[i] = np.mean([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']]) 
    i +=1

T_mean_exp_80 = 0.5*T_mean_exp1 + 0.5*T_mean_exp2
time_vect_meanT = (data["Chrono"][index_min:index_max] - t_min)/3600
    


name = 'T_loss75_3424_70hours'
path = '..\\data\\Simulations\\VELIS_valid\\'

results = wh.procF.open_results(name, path = path)
V = results.Total_volume
E_min = results.E_min
E_max = results.E_max

E_sim = abs(results.E_power - results.E_power[0])

T_mean_sim80 = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    
    T_mean_sim80[k] = 0.5*np.mean(results.T_record1[k]) + 0.5*np.mean(results.T_record2[k]) -273.15



fig, ax = plt.subplots(figsize=(5.5/1.5,4.3/1.5),constrained_layout=True)   

plt.rcParams.update({'font.size':'16'})
params = {
          "text.usetex" : True,
          "font.family" : "cm"}
plt.rcParams.update(params)

plt.grid()
plt.xlabel('Time [h]', fontsize=16)
plt.ylabel('Avg temperature [$^\circ$C]',fontsize=16)  

plt.plot(np.array(results.time_vect)/3600, T_mean_sim80 , color = 'k',  linewidth = 1.5,zorder=1)  
plt.scatter(time_vect_meanT[::20000], T_mean_exp_80[::20000], marker="o", s=36, color = 'r',zorder=2)  
plt.xlim(0,70)
plt.xticks((0,10,20,30,40,50,60,70))
plt.ylim(30,80)
plt.yticks((30,40,50,60,70,80))
# fmt = mdates.DateFormatter('%H:%M')
# ax.xaxis.set_major_formatter(fmt)

# Error 
t_sim = np.array(results.time_vect)
T_sim = T_mean_sim80
t_exp = time_vect_meanT*3600
T_exp = T_mean_exp_80

# Interpolate the simulated data over the experimental time points
interpolator = interp1d(t_sim, T_sim, kind='linear', fill_value='extrapolate')
T_sim_interpolated = interpolator(t_exp)

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(T_sim_interpolated - T_exp))
print(f"Mean Absolute Error heat losses: {mae}")

#%% Time variation with flow rate -- temperature profile 

"""Read TDMS File"""

data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_E\8minutes_discharge.TDMS",0)


start_index = 30
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 
data["Chrono"] = data["Chrono"]  - start_index
data['m_dot_avg'] = [ser['m_dot_w'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
data['T_w_in_avg'] = [ser['T_w_in'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
m_dot_avg = np.mean(data['m_dot_avg'])
T_w_in_avg = np.mean(data['T_w_in_avg'])

name = 'discharge_85gps'
results = wh.procF.open_results(name, path = path)


t_minute = 180
t = t_minute*60
exp_data = {}
exp_data['h'] = np.linspace(10, 90,9)
index = (data['Chrono'] - t).abs().idxmin()
row = data.loc[index]
exp_data['T1'] = np.array([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 
exp_data['T2'] = np.array([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']])       
exp_data['T_avg'] = 0.5*np.mean(exp_data['T1']) + 0.5*np.mean(exp_data['T2'])
index_sim = np.argmin(abs(np.array(results.time_vect) - t))
results.plotTPointsVelis(index = index_sim,exp_data = exp_data)



#%% Time variation of the power consumption 


"""Read TDMS File"""
data = TDMS_to_CSV("..\\data\\Experimental\\Experimental_results_E\\24h_scenario.TDMS",0)
start_index = 20
data["Chrono"] = data["Time"] - data["Time"][0]
data["Chrono"] = data['Chrono'].dt.total_seconds() 
data["Chrono"] = data["Chrono"]  - start_index


"""Read simulation file"""


name = '24h_scenario_bis'
path = '..\\data\\Simulations\\VELIS_valid\\'
results = wh.procF.open_results(name, path = path)
    

W_dot_el = np.array(results.W_dot_cons_1) + np.array(results.W_dot_cons_2)



t_min = 0
t_max = 24*3600+100000
index_min = (data['Chrono'] - t_min).abs().idxmin()
index_max = (data['Chrono'] - t_max).abs().idxmin()
index_vect = np.arange(index_min, index_max,1)
T_mean_exp1 = np.zeros(len(index_vect))
T_mean_exp2 = np.zeros(len(index_vect))

i=0
for k in index_vect:
    row = data.loc[k]
    T_mean_exp1[i] = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 
    T_mean_exp2[i] = np.mean([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']]) 
    i +=1

T_mean_exp_80 = 0.5*T_mean_exp1 + 0.5*T_mean_exp2

T_mean_sim80 = np.zeros(len(results.T_record1))
for k in range(len(results.T_record1)):
    
    T_mean_sim80[k] = 0.5*np.mean(results.T_record1[k]) + 0.5*np.mean(results.T_record2[k]) -273.15


time_vect = data["Chrono"]/3600


data['m_dot_w']  = [ser['m_dot_w'] - 0.1*ser['m_dot_w'] if ser['m_dot_w'] > 10 else 0  for idx, ser in data.iterrows()]
data['T_w_in_avg'] = [ser['T_w_in'] if ser['m_dot_w'] > 10 else np.nan  for idx, ser in data.iterrows()]
data['W_dot_el_avg1'] = [ser['Power_el_res1'] - 100 if ser['Power_el_res1'] > 10 else np.nan  for idx, ser in data.iterrows()]
data['W_dot_el_avg2'] = [ser['Power_el_res2'] - 100 if ser['Power_el_res2'] > 10 else np.nan  for idx, ser in data.iterrows()]

W_dot_avg = 0.5*np.mean(data['W_dot_el_avg1']) + 0.5*np.mean(data['W_dot_el_avg2'])
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
plt.ylim(20, 100)
# plt.xlim(0, 25)
plt.yticks([20,30, 40,50,60,70,80,90,100])   


ax.plot(np.array(results.time_vect)/3600, T_mean_sim80, c = u'#ff7f0e' , linewidth = 2,zorder=1,label =  'Simulation') 
ax.scatter(time_vect[::1000], T_mean_exp_80[::1000], marker="o", s=36, color = 'r',zorder=2, label = 'Experimental')  
ax2=ax.twinx()   
ax2.set_ylabel('Mass flow rate [g/s]' ,fontsize=fontsize, color=  u'#1f77b4') 
ax2.set_ylim(0, 100) 
# ax2.set_yticks([70,74,78,82,86,90])
plt.grid()
ax2.plot(time_vect,data['m_dot_w'] , c = u'#1f77b4', linewidth = 2) 

ax.legend(bbox_to_anchor=(0.72,1.02), ncol=1, fontsize=12)
del results.time_vect[-1]    
    
#def plot_time_variation(data, t_start, t_stop):
xdim = 5.5+1
ydim = 4.3
labelsize = 18



row = data.loc[0]
T_mean_exp1 = np.mean([row['T_tank1_1'], row['T_tank1_2'], row['T_tank1_3'], row['T_tank1_4'], row['T_tank1_5'], row['T_tank1_6'], row['T_tank1_7'], row['T_tank1_8'], row['T_tank1_9']]) 
T_mean_exp2 = np.mean([row['T_tank2_1'], row['T_tank2_2'], row['T_tank2_3'], row['T_tank2_4'], row['T_tank2_5'], row['T_tank2_6'], row['T_tank2_7'], row['T_tank2_8'], row['T_tank2_9']]) 
T_mean_exp_80 = 0.5*T_mean_exp1 + 0.5*T_mean_exp2

##### power, temperature at sensor

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
plt.ylim(-100, 3000)
# plt.xlim(0, 25)
plt.yticks([0, 500,1000, 1500,2000,2500,3000])   

data['power_tot'] = data['Power_el_res1'] + data['Power_el_res2'] - 100
# l1 = ax.plot(time_vect[::50], data['power_tot'][::50] , c = u'#ff7f0e' , linewidth = 2)   

ax.plot(np.array(results.time_vect)/3600, W_dot_el , c = u'#ff7f0e' , linewidth = 2,zorder=1,label =  'Simulation') 
ax.scatter(time_vect[::2000], data['power_tot'][::2000], marker="o", s=36, color = 'r',zorder=2, label = 'Experimental')  
ax2=ax.twinx()   
ax2.set_ylabel('Mass flow rate [g/s]' ,fontsize=fontsize, color=  u'#1f77b4') 
ax2.set_ylim(0, 100) 
# ax2.set_yticks([70,74,78,82,86,90])
plt.grid()
ax2.plot(time_vect,data['m_dot_w'] , c = u'#1f77b4', linewidth = 2) 

# labels = ['Simulation','Experimental']  
ax.legend(bbox_to_anchor=(0.72,1.02), ncol=1, fontsize=12)
# labels = [ 'Power tank 1','Power tank 2','Mass flow rate']  
# plt.legend(lns, labels,fontsize=12)
# labels = [ 'Oil Circulation Ratio','Pressure ratio']  
# plt.legend(lns, labels,fontsize=14, loc='best')





