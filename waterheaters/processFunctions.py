# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:46:17 2023

@author: nicol
"""

from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt
import time as counter
from scipy.interpolate import interp1d
import pickle
import os

class WaterHeater():
    def __init__(self, Volume = None, Diameter = None, Height = None, EWH = False, HPWH = False, double = False):
        
        """
        Initialization of the WaterHeater
        
        Inputs:
            fluid: type of fluid used in the storage
            Volume: volume of the storage (of one tank only)
            Diameter: diameter of the storage
            Height: height of the storage
            EWH or HPWH: type of heating system
            refrigerant: type of refrigerant required for HPWH
            double: True for the VELIS (two storage tanks)
        """
        self.fluid = 'water'
        self.Volume = Volume
        self.Diameter = Diameter
        if self.Volume != None:
            self.Height = self.Volume*4/np.pi/self.Diameter**2 # height of the storage
        else :
            self.Height = Height
            self.Volume = self.Height*np.pi*self.Diameter**2/4
        self.EWH = EWH # check if the heater is electric
        self.HPWH = HPWH # check if the heater is a heat pump 
        # Without EWH and HPWH, the tank is just a storage tank
        # if isinstance(refrigerant, str):
        #     self.refrigerant = refrigerant
        
        self.double  = double 
        if self.double == True: 
            self.tank_number = 2

        else: 
            self.tank_number = 1

        self.Total_volume = self.Volume*self.tank_number


    def Heating_system(self, z_control = 0.0, z_init_E = 0.0, z_init_HP = 0.0, height_E = 0.0, height_HP = 0.0, Q_dot_peak_E = 0):
        """
        Initialization of the heating system if required 
        
        Inputs:
            E = electric
            HP = heat pump
            z_init: height were the heating system starts
            height: height of the heating system 
            Q_dot_peak: peak power of the electrical resistor 
            
        """
        self.param_heating = {'z_control': z_control, 'z_init_E' : z_init_E, 'z_init_HP' : z_init_HP, 'height_E' : height_E,
                               'height_HP' : height_HP, 'Q_dot_peak_E' : Q_dot_peak_E}
        
    def Model_parameters(self, h_amb = 0, delta = 10000, H_mix = 0, V_s = 10/1e6, W_dot_el_basis = 50):

        """
        Initialization of the model parameters
        
        Inputs:
            h_amb: heat transfer coefficient with the ambiance (empirical)
            s: averaged boudary temperature correction coefficient (empirical)
            delta: coefficient taking into account the reversing effect du to density difference between two adjacent layers
            H_mix: height of mixing in meters to ditribute the inlet mass flow rate
            
        """
        if self.HPWH == True:
            self.refrigerant = 'propane' # only used in the case where a Heat Pump water heater is used
        else:
            self.refrigerant = None 
        self.MParam = {'h_amb' : h_amb, 'delta' : delta, 'H_mix' : H_mix, 'V_s' : V_s, 'W_dot_el_basis' : W_dot_el_basis}       


        
    def MN_model_init(self, nx, T_vect_init_1, T_vect_init_2 = np.array([])):
        
        """
        Function used to initiate the MN model
        
        Inputs:
            nx: number of zones
            T_vect_init_1: initial temperature vector
            T_vect_init_2: When a double-configuration is used, another temperature vector is necessary
        """
        if self.double == True and T_vect_init_2.size == 0:
            print('When a double-configuration is used, another temperature vector is necessary')
        
        # Layer dictionnary initialization
        self.layer = {}
        self.layer['nx'] = nx
        self.layer["H"] = np.full(nx, self.Height/nx)
        self.layer["V"] = np.full(nx, self.Volume/nx)
        self.layer["A_wall"] = self.layer["H"]*np.pi*self.Diameter
        self.layer["A_wall_amb"] = self.layer["A_wall"].copy()
        self.layer["A_wall_amb"][0] = self.layer["A_wall"][0] + np.pi*self.Diameter**2/4 # addition of the top area
        self.layer["A_wall_amb"][-1] = self.layer["A_wall"][-1] + np.pi*self.Diameter**2/4 # addition of the bottom area
        # A_wall_amb is used for the ambiance losses (takes the top and the bottom area into account)
        # A_wall is only used for the heat exchange with the refrigerant 
        
        # Calculation of the distance between the center of the layers for the diffusivity equation
        self.dx = np.zeros(nx-1)
        for i in range(0, nx-1):
            # self.H_cumul[i] = self.H_cumul[i-1] + self.layer["H"][i]/2 + self.layer["H"][i]/2
            self.dx[i] = self.layer["H"][i]/2 + self.layer["H"][i]/2
            
          
        # Initilization of transport properties from the initial vector distribution
        P_sto = 500000 # water pressure assumption
        cp_sto = PropsSI( 'CPMASS',  'T', np.mean(np.concatenate((T_vect_init_1, T_vect_init_2), axis=None)), 'P', P_sto, self.fluid)
        rho_sto = PropsSI( 'D',  'T', np.mean(np.concatenate((T_vect_init_1, T_vect_init_2), axis=None)), 'P', P_sto, self.fluid)
        k_sto = PropsSI( 'CONDUCTIVITY',  'T', np.mean(np.concatenate((T_vect_init_1, T_vect_init_2), axis=None)), 'P',  P_sto, self.fluid)
        alpha_sto = k_sto/(rho_sto * cp_sto)*0
        
        self.prop_dict = {'cp_sto' : cp_sto, 'rho_sto' : rho_sto, 'k_sto': k_sto, 'alpha_sto': alpha_sto}
        # Calculation of the storage mass
        self.layer["m"] = self.layer["V"]*self.prop_dict['rho_sto']   
        
        ## Definition of the layers where the heating system is used
        self.k_EWH = np.zeros(nx)
        self.k_HPWH = np.zeros(nx)
        
        if self.Height < (self.param_heating['z_init_E'] + self.param_heating['height_E']) or self.Height < (self.param_heating['z_init_HP'] + self.param_heating['height_HP']):
            print("Bad dimensions of the heating system")
        
        if self.EWH == True or self.HPWH == True:
            h_cumul = 0
            for z in range(nx):
                if (h_cumul >= self.param_heating['z_init_E']) and (h_cumul < (self.param_heating['z_init_E'] + self.param_heating['height_E'])):
                    self.k_EWH[z] = 1
                if (h_cumul >= self.param_heating['z_init_HP']) and (h_cumul < (self.param_heating['z_init_HP'] + self.param_heating['height_HP'])):
                    self.k_HPWH[z] = 1
                h_cumul += self.layer['H'][z]
        
        
        # Number of nodes for the water injection
        self.k_l = 1
        while self.MParam['H_mix'] > self.k_l * self.layer['H'][0]:
            self.k_l += 1
            
        # fraction of height for the control 
        self.control_ratio = 1 - self.param_heating['z_control']/self.Height
        
        
        # Inversion of the vector as it starts from 0 that is the top of the boiler
        self.k_EWH = np.flip(self.k_EWH)
        self.k_HPWH = np.flip(self.k_HPWH)
        # print(np.tile(self.layer["m"], self.tank_number), np.concatenate((T_vect_init_1, T_vect_init_2)))
        # Initialization of the working variables
        self.time_vect = [0]
        #temperature vector
        self.T_record1 = [T_vect_init_1]
        self.T_record2 = [T_vect_init_2]
        self.E_min = sum(np.tile(self.layer["m"], self.tank_number)*cp_sto*(273.15 + 15))
        self.E_max = sum(np.tile(self.layer["m"], self.tank_number)*cp_sto*(273.15 + 90))
        self.E_temp = [sum(np.tile(self.layer["m"], self.tank_number)*cp_sto*np.concatenate((T_vect_init_1, T_vect_init_2), axis=None))]
        self.E_power = [sum(np.tile(self.layer["m"], self.tank_number)*cp_sto*np.concatenate((T_vect_init_1, T_vect_init_2), axis=None))]
        self.SOC = [(self.E_temp[0] - self.E_min)/(self.E_max - self.E_min)]
        self.Q_dot_ref = []
        self.Q_dot_water_used = []
        self.Q_dot_amb = []
        self.Q_dot_E = []
        self.W_dot_cons_HP = []
        self.W_dot_cons_1 = []
        self.W_dot_cons_2 = []
        self.W_dot_cons_tot = []
        self.T_w_out = []
        self.m_dot_w = []
        self.COP = []
        self.Q_dot_ref_vect = []
        self.perf_solving = []
        self.perf_building = []
        
    def MN_model(self, dt, time, m_dot, T_w_supply, T_amb, T_SP, T_ext = 273.15):
        """
        Function used to model the Waterheater using Modal nodes (division into nx zones)     
        
        Inputs:
            
            dt: time step - s
            time: maximum time - s
            T_vect_init: initial temperature vector - K
            m_dot: mass flow rate flowing in and out of the storage tank - kg/s
            T_w_supply: temperature of the supply water - K
            T_amb: ambiant temperature - K
            T_SP: temperature set point of the storage tank - K
            T_ext: external temperature for the HP - K
        """
        # new sequence of sim launched
        new_sim = True
        
        # Matrices initialization
        
        # Initiliaze the matrix size
        self.A1 = np.zeros([self.layer['nx'],self.layer['nx']])
        self.B1 = np.zeros([self.layer['nx'],self.layer['nx']])
        self.C1 = np.zeros([self.layer['nx']])
        K = np.zeros([self.layer['nx']])
        
        if self.double == True:
            # Initiliaze the matrix size
            self.A2 = np.zeros([self.layer['nx'],self.layer['nx']])
            self.B2 = np.zeros([self.layer['nx'],self.layer['nx']])
            self.C2 = np.zeros([self.layer['nx']])
            K2 = np.zeros([self.layer['nx']])            
        # Surface between two layers 
        A_bound = self.Diameter**2*np.pi/4
        

            
        # Case where an electrical resistor is used 
        Q_dot_E_layer = 0
        if self.EWH == True:
            Q_dot_E_layer = self.param_heating['Q_dot_peak_E']/sum(self.k_EWH)
            
        Q_dot_E_layer_vect = self.k_EWH*Q_dot_E_layer

        
        # Ambiance heat transfer
        h_amb = self.MParam['h_amb'] #W/(m^2 K)
        
        ## Resolution 
        # Resolution of the equation of type AT + BT0 + C = 0
        
        # Initial temperature distribution
        
        T_1 = self.T_record1[-1]
        T_2 = self.T_record2[-1]
        
        
        # Function to check for the temperature inversion phenomenon (if T[i] > T[i-1] + tol, the inversion should happen)
        T_diff_tol = 0 # K
        def is_increasing(vector):
            for i in range(1, len(vector)):
                if vector[i] > vector[i-1] + T_diff_tol:
                    return True
            return False
        
        switch1 = 0
        switch2 = 0
        for t in range(int(round(time/dt,0))):
            
            
            # Check if the system is heating or not (condition inside the function)
            switch1_pre = switch1

            if self.double == False:
                switch1, _ = self.control(self.layer['nx'], T_1, T_SP, switch1_pre)
            else: 
                switch2_pre = switch2
                switch1, switch2 = self.control(self.layer['nx'], T_1, T_SP, switch1_pre, switch2_pre = switch2_pre, T2 = T_2)
                
            # increasing to check if a layer is not warmer below a hot layer --> K_i changing
            # switch if a change is noticed the matrices need to be changed --> switch changing
            # new_sim as calculation is needed for the first sim
            # switch == 1 if heating K_i will automatically change
            
            if is_increasing(T_1) or switch1 != switch1_pre or new_sim or switch1 == 1:
                tic_building_1 = counter.perf_counter()
                #not(hasattr(self, 'A')) condition for the creation of the matrix, not in use anymore

                # water temperature 
                # Check if any values in self.k_HPWH are equal to 1
                if self.HPWH == True:
                    W_dot_cons_HP, Q_dot_cd, COP = self.HP_cycle(T_1, switch1, T_ext)
                    self.COP.append(COP)
                    Q_dot_cd_i = Q_dot_cd/sum(self.k_HPWH)
                else :
                    W_dot_cons_HP = 0
                    Q_dot_cd_i = 0
                    self.COP.append(None)    

                 
                # Loop on the storage size to fill in the matrices
                for i in range(0,self.layer['nx']):

                    if i != self.layer['nx']-1:
                        if T_1[i + 1] > T_1[i] + T_diff_tol:
                            K[i] = self.prop_dict['k_sto']*self.MParam['delta']#*abs(T[i + 1] - T[i])
                        else:
                            K[i] = self.prop_dict['k_sto']
                        
                    if i == 0: 
                        self.A1[0][0] = self.layer["m"][0]*self.prop_dict['cp_sto']/dt + m_dot*self.prop_dict['cp_sto'] + K[0]*A_bound/self.dx[0] - self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']/self.dx[0]**2 \
                            + h_amb*self.layer["A_wall_amb"][0] 
                            
                        self.A1[0][1] = -m_dot*self.prop_dict['cp_sto'] + self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']*(1/self.dx[0]**2 + 1/self.dx[1]**2) - K[0]*A_bound/self.dx[0]
                        
                        self.A1[0][2] = - self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']/self.dx[1]**2
                        
                        self.B1[0][0] = - self.layer["m"][0]*self.prop_dict['cp_sto']/dt
                        
                        self.C1[0] = -h_amb*self.layer["A_wall_amb"][0]*T_amb - switch1*self.k_EWH[0]*Q_dot_E_layer_vect[0] \
                            - switch1*self.k_HPWH[0]*Q_dot_cd_i
                            
                    elif i > self.layer['nx']-1 - self.k_l: 
                        
                        nx_new = self.layer['nx']-1
                        ## case of the last cell
                        if i == self.layer['nx']-1:
                            
                            self.A1[nx_new][nx_new - 2] = - self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']/self.dx[nx_new-2]**2 
                                
                            self.A1[nx_new][nx_new - 1] = self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']*(1/self.dx[nx_new-1]**2 + 1/self.dx[nx_new-2]**2) - K[nx_new-1]*A_bound/self.dx[nx_new-1]
                            
                            self.A1[nx_new][nx_new] = self.layer["m"][nx_new]*self.prop_dict['cp_sto']/dt + m_dot/self.k_l*self.prop_dict['cp_sto'] - self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']/self.dx[nx_new-1]**2 + h_amb*self.layer["A_wall_amb"][nx_new] \
                               + K[nx_new-1]*A_bound/self.dx[nx_new-1]
                                
                            self.B1[nx_new][nx_new] = - self.layer["m"][nx_new]*self.prop_dict['cp_sto']/dt
                            
                            self.C1[nx_new] = -h_amb*self.layer["A_wall_amb"][nx_new]*T_amb - switch1*self.k_EWH[nx_new]*Q_dot_E_layer_vect[nx_new] \
                                - switch1*self.k_HPWH[nx_new]*Q_dot_cd_i- m_dot/self.k_l*self.prop_dict['cp_sto']*T_w_supply
                          ## case of the remaining cells with injection     
                        else:
                            
                            frac_in = (nx_new - i)/self.k_l  # used for the fraction of mass flow rate going inside the cell from the previous cell 
                            frac_out = (nx_new - i + 1)/self.k_l  # used for the fraction of mass flow rate going outside the cell in the next cell 
                            
                            
                            self.A1[i][i - 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i-1]**2 - K[i-1]*A_bound/self.dx[i-1]
                                
                            self.A1[i][i + 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i]**2 - m_dot*frac_in*self.prop_dict['cp_sto'] - K[i]*A_bound/self.dx[i]
                            
                            self.A1[i][i] = self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']*(1/self.dx[i]**2 + 1/self.dx[i-1]**2) \
                                + self.layer["m"][i]*self.prop_dict['cp_sto']/dt + m_dot*frac_out*self.prop_dict['cp_sto'] + h_amb*self.layer["A_wall_amb"][i] \
                                 + K[i]*A_bound/self.dx[i] + K[i-1]*A_bound/self.dx[i-1]
                            
                            self.B1[i][i] = - self.layer["m"][i]*self.prop_dict['cp_sto']/dt
                            
                            self.C1[i] = -h_amb*self.layer["A_wall_amb"][i]*T_amb - switch1*self.k_EWH[i]*Q_dot_E_layer_vect[i] \
                                - switch1*self.k_HPWH[i]*Q_dot_cd_i - m_dot/self.k_l*self.prop_dict['cp_sto']*T_w_supply
                            
                            
                    else: 
                        self.A1[i][i - 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i-1]**2 - K[i-1]*A_bound/self.dx[i-1]
                            
                        self.A1[i][i + 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i]**2 - m_dot*self.prop_dict['cp_sto'] - K[i]*A_bound/self.dx[i]
                        
                        self.A1[i][i] = self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']*(1/self.dx[i]**2 + 1/self.dx[i-1]**2) \
                            + self.layer["m"][i]*self.prop_dict['cp_sto']/dt + m_dot*self.prop_dict['cp_sto'] + h_amb*self.layer["A_wall_amb"][i] \
                             + K[i]*A_bound/self.dx[i] + K[i-1]*A_bound/self.dx[i-1]
                        
                        self.B1[i][i] = - self.layer["m"][i]*self.prop_dict['cp_sto']/dt
                        
                        self.C1[i] = -h_amb*self.layer["A_wall_amb"][i]*T_amb - switch1*self.k_EWH[i]*Q_dot_E_layer_vect[i] \
                            - switch1*self.k_HPWH[i]*Q_dot_cd_i
                        
                toc_building_1 = counter.perf_counter()
                self.perf_building.append(toc_building_1 - tic_building_1)
            
            # Resolution process   
            tic_solving_1 = counter.perf_counter()
            T0 = T_1
            T_1 =  np.dot(np.linalg.inv(self.A1),  - np.transpose((np.dot(self.B1,T0) + self.C1)))
            toc_solving_1 = counter.perf_counter()
            self.perf_solving.append(toc_solving_1 - tic_solving_1)
            
            # Vector of power exchanges
            Q_dot_ref_vect = switch1*self.k_HPWH*Q_dot_cd_i
            Q_dot_amb_vect = -h_amb*self.layer["A_wall_amb"]*(T_1 - T_amb)
            Q_dot_E_vect = switch1*Q_dot_E_layer_vect
            self.W_dot_cons_1.append(sum(Q_dot_E_vect))
            E_temp = sum(self.layer["m"]*self.prop_dict['cp_sto']*T_1)
            
            if self.double == True:
                
                # Same conditions as in the first tank except for the mass flow rate that has an influence here
                if is_increasing(T_2) or switch2 != switch2_pre or new_sim or switch2 == 1 or m_dot != 0:
                    tic_building_2 = counter.perf_counter()
                    #not(hasattr(self, 'A')) condition for the creation of the matrix, not in use anymore

                    # Commented as double tank storage does not apply with heat pump heating system
                    # water temperature 
                    # Check if any values in self.k_HPWH are equal to 1
                    # if self.HPWH == True:
                    #     # Get the first matching value from the T array
                    #     # Highest temperature where the water is heated (pinch point at Q = 1)
                    #     T_w_out_ev = max(T_2[self.k_HPWH == 1])
                    #     pp_cd = 5 #K
                    #     delta = T_amb - 273.15
                    #     pp_cd = 5 + delta #K
                    #     T_ref = T_w_out_ev + pp_cd
                    #     # Chose the pressure reference with a pinch point of 5K at a quality Q = 1
                    #     P_ref = PropsSI( 'P',  'T', T_ref, 'Q', 0.5, self.refrigerant)
                    # else:
                    #     T_ref = 273.15

                    # Loop to create the matrices of the second tank 
                    T_supply_tank2 = T_1[0]

                    for i in range(0,self.layer['nx']):
                        
                        if i != self.layer['nx']-1:
                            if T_2[i + 1] > T_2[i] + T_diff_tol:
                                K2[i] = self.prop_dict['k_sto']*self.MParam['delta']#*abs(T[i + 1] - T[i])
                            else:
                                K2[i] = self.prop_dict['k_sto']
                            
                        if i == 0: 
                            self.A2[0][0] = self.layer["m"][0]*self.prop_dict['cp_sto']/dt + m_dot*self.prop_dict['cp_sto'] + K2[0]*A_bound/self.dx[0] - self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']/self.dx[0]**2 \
                                + h_amb*self.layer["A_wall_amb"][0] 
                                
                            self.A2[0][1] = -m_dot*self.prop_dict['cp_sto'] + self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']*(1/self.dx[0]**2 + 1/self.dx[1]**2) - K2[0]*A_bound/self.dx[0]
                            
                            self.A2[0][2] = - self.prop_dict['alpha_sto']*self.layer["m"][0]*self.prop_dict['cp_sto']/self.dx[1]**2
                            
                            self.B2[0][0] = - self.layer["m"][0]*self.prop_dict['cp_sto']/dt
                            
                            self.C2[0] = -h_amb*self.layer["A_wall_amb"][0]*T_amb - switch2*self.k_EWH[0]*Q_dot_E_layer_vect[0] \
                               
                        
                        elif i > self.layer['nx']-1 - self.k_l: 
                            nx_new = self.layer['nx']-1
                            ## case of the last cell
                            if i == self.layer['nx']-1:
                                self.A2[nx_new][nx_new - 2] = - self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']/self.dx[nx_new-2]**2 
                                    
                                self.A2[nx_new][nx_new - 1] = self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']*(1/self.dx[nx_new-1]**2 + 1/self.dx[nx_new-2]**2) - K2[nx_new-1]*A_bound/self.dx[nx_new-1]
                                
                                self.A2[nx_new][nx_new] = self.layer["m"][nx_new]*self.prop_dict['cp_sto']/dt + m_dot/self.k_l*self.prop_dict['cp_sto'] - self.prop_dict['alpha_sto']*self.layer["m"][nx_new]*self.prop_dict['cp_sto']/self.dx[nx_new-1]**2 + h_amb*self.layer["A_wall_amb"][nx_new] \
                                     + K2[nx_new-1]*A_bound/self.dx[nx_new-1]
                                    
                                self.B2[nx_new][nx_new] = - self.layer["m"][nx_new]*self.prop_dict['cp_sto']/dt
                                
                                self.C2[nx_new] = -h_amb*self.layer["A_wall_amb"][nx_new]*T_amb - switch2*self.k_EWH[nx_new]*Q_dot_E_layer_vect[nx_new] \
                                     - m_dot/self.k_l*self.prop_dict['cp_sto']*T_supply_tank2
                            
                    
                              ## case of the remaining cells with injection     
                            else:
                                
                                frac_in = (nx_new - i)/self.k_l  # used for the fraction of mass flow rate going inside the cell from the previous cell 
                                frac_out = (nx_new - i + 1)/self.k_l  # used for the fraction of mass flow rate going outside the cell in the next cell 
                                
                                
                                self.A2[i][i - 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i-1]**2 - K2[i-1]*A_bound/self.dx[i-1]
                                    
                                self.A2[i][i + 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i]**2 - m_dot*frac_in*self.prop_dict['cp_sto'] - K2[i]*A_bound/self.dx[i]
                                
                                self.A2[i][i] = self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']*(1/self.dx[i]**2 + 1/self.dx[i-1]**2) \
                                    + self.layer["m"][i]*self.prop_dict['cp_sto']/dt + m_dot*frac_out*self.prop_dict['cp_sto'] + h_amb*self.layer["A_wall_amb"][i] \
                                   + K2[i]*A_bound/self.dx[i] + K2[i-1]*A_bound/self.dx[i-1]
                                
                                self.B2[i][i] = - self.layer["m"][i]*self.prop_dict['cp_sto']/dt
                                
                                self.C2[i] = -h_amb*self.layer["A_wall_amb"][i]*T_amb - switch2*self.k_EWH[i]*Q_dot_E_layer_vect[i] \
                                     - m_dot/self.k_l*self.prop_dict['cp_sto']*T_supply_tank2
                                
                  
                        else: 
                            self.A2[i][i - 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i-1]**2 - K2[i-1]*A_bound/self.dx[i-1]
                                
                            self.A2[i][i + 1] = - self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']/self.dx[i]**2 - m_dot*self.prop_dict['cp_sto'] - K2[i]*A_bound/self.dx[i]
                            
                            self.A2[i][i] = self.prop_dict['alpha_sto']*self.layer["m"][i]*self.prop_dict['cp_sto']*(1/self.dx[i]**2 + 1/self.dx[i-1]**2) \
                                + self.layer["m"][i]*self.prop_dict['cp_sto']/dt + m_dot*self.prop_dict['cp_sto'] + h_amb*self.layer["A_wall_amb"][i] \
                              + K2[i]*A_bound/self.dx[i] + K2[i-1]*A_bound/self.dx[i-1]
                            
                            self.B2[i][i] = - self.layer["m"][i]*self.prop_dict['cp_sto']/dt
                            
                            self.C2[i] = -h_amb*self.layer["A_wall_amb"][i]*T_amb - switch2*self.k_EWH[i]*Q_dot_E_layer_vect[i] \
                
                    toc_building_2 = counter.perf_counter() 
                    self.perf_building.append(toc_building_2 - tic_building_2)
                
                # Resolution process 
                tic_solving_2 = counter.perf_counter()
                T0 = T_2
                T_2 =  np.dot(np.linalg.inv(self.A2),  - np.transpose((np.dot(self.B2,T0) + self.C2)))            
                toc_solving_2 = counter.perf_counter()
                self.perf_solving.append(toc_solving_2 - tic_solving_2)

            # Vector of power exchanges
            self.Q_dot_ref_vect.append(Q_dot_ref_vect)

            if self.tank_number == 2:
                Q_dot_amb_vect = Q_dot_amb_vect - h_amb*self.layer["A_wall_amb"]*(T_2 - T_amb)
                Q_dot_E_vect = Q_dot_E_vect + switch2*Q_dot_E_layer_vect
                self.W_dot_cons_2.append(sum(switch2*Q_dot_E_layer_vect))
                T_out = T_2[0]
                E_temp += sum(self.layer["m"]*self.prop_dict['cp_sto']*T_2)
            else:
                T_out = T_1[0]
            self.time_vect.append(self.time_vect[-1] + dt)
            self.Q_dot_amb.append(sum(Q_dot_amb_vect))
            self.Q_dot_ref.append(sum(Q_dot_ref_vect))
            self.Q_dot_E.append(sum(Q_dot_E_vect))
            self.Q_dot_water_used.append(m_dot*self.prop_dict['cp_sto']*(T_out - T_w_supply))
            self.m_dot_w.append(m_dot)
            
            # Temperature of the water flowing out
            self.T_w_out.append(T_out)

            

                    
            self.W_dot_cons_HP.append(W_dot_cons_HP)
            self.W_dot_cons_tot.append(self.HPWH*self.W_dot_cons_HP[-1] + self.EWH*self.Q_dot_E[-1])
            
            
            ## Energy conservation
            # From the storage point of view (temperature)
            self.E_temp.append(E_temp)
            # From the power gain and loss 
            self.E_power.append(self.E_power[-1] + self.Q_dot_ref[-1]*dt + self.Q_dot_amb[-1]*dt + self.Q_dot_E[-1]*dt - self.Q_dot_water_used[-1]*dt)
            # State of charge
            self.SOC.append((self.E_temp[-1] - self.E_min)/(self.E_max - self.E_min))
            ## New temperature vector recording
            self.T_record1.append(T_1)
            self.T_record2.append(T_2)
            new_sim = False
            

            
    def control(self, nx, T, T_SP, switch_pre, switch2_pre = 0, T2 = 0):
        
        """
        Control strategy of the water heater
        Strat: 
        The system stops heating when the set point is exceeded by a temperature difference DT_high
        The system starts heating when the control temperatrure is DT_low below the set point
        switch2 always has the priority on switch 1
        
        """
        DT_high = -1
        # DT_high = -3

        ## Control probes
        DT_low = 7
        # DT_low = 25
        hyst = DT_low + DT_high
        T_SP = T_SP + DT_high
        
        if self.double == False:
            T_probe = T[int(np.round(nx*(self.control_ratio)))]
            
            if T_probe >= T_SP:
                switch1 = 0
            elif T_probe <= T_SP - hyst:
                switch1 = 1
            else:
                switch1 = switch_pre
        
            switch2 = 0
        else: 
            T_probe1 = T[int(np.round(nx*(self.control_ratio)))]
            T_probe2 = T2[int(np.round(nx*(self.control_ratio)))]
            if T_probe1 >= T_SP:
                switch1 = 0
            elif T_probe1 <= T_SP - hyst:
                switch1 = 1
            else:
                switch1 = switch_pre
            if T_probe2 >= T_SP:
                switch2 = 0
            elif T_probe2 <= T_SP - hyst:
                switch2 = 1
            else:
                switch2 = switch2_pre
            if switch1 == 1 and switch2 == 1: 
                switch1 = 0

        return switch1, switch2
    
    def HP_cycle(self,T_1, switch1, T_ext):
        
        """Function analyzing the heat pump cycle, allowing to retrieve the compressor electrical consumption"""
        
        # Get the first matching value from the T array
        # Highest temperature where the water is heated (pinch point = 0 )
        T_w_max = max(T_1[self.k_HPWH == 1])
        SC = 5  #K
        T_sat_cd = T_w_max + SC
        # Subcooling of 5K used to get the saturation pressure
        P_cd = PropsSI( 'P',  'T', T_sat_cd, 'Q', 0.5, self.refrigerant)
        
        
        
        # Heat source ref pressure
        pp_ev = 10 #K
        SH = 5 #K
        T_sat_ev = T_ext - pp_ev
        P_ev = PropsSI( 'P',  'T', T_sat_ev, 'Q', 0.5, self.refrigerant)
        
        # Compressor analysis
        h_su_cp = PropsSI( 'H',  'T', T_sat_ev  + SH, 'P', P_ev, self.refrigerant)
        rho_su_cp = PropsSI( 'D',  'T', T_sat_ev  + SH, 'P', P_ev, self.refrigerant)
        s_su_cp = PropsSI( 'S',  'T', T_sat_ev  + SH, 'P', P_ev, self.refrigerant)
        h_is_cp = PropsSI( 'H',  'S', s_su_cp, 'P', P_cd, self.refrigerant)
        w_is = h_is_cp - h_su_cp
        
        f = 50*0.97 # Hz
        V_s = 5/1e6 # cubic meter
        V_dot_cp = f*V_s # cm per second
        m_dot_cp_th = rho_su_cp*V_dot_cp # kg/second
        eps_v = 0.8
        eps_is = 0.75
        m_dot_cp = m_dot_cp_th * eps_v

        w = w_is/eps_is
        if switch1 != 0:
            W_dot_cp_el = w*m_dot_cp_th + self.MParam['W_dot_el_basis']
        else: 
            W_dot_cp_el = 0
            
        
        # Cd analysis
        h_su_cd = h_su_cp + w
        h_ex_cd = PropsSI( 'H',  'T', T_w_max, 'P', P_cd, self.refrigerant)
        Q_dot_cd = m_dot_cp*(h_su_cd - h_ex_cd)
           
        COP = Q_dot_cd/(W_dot_cp_el+1e-6)

        return W_dot_cp_el, Q_dot_cd, COP
            
        
    def plotTPoints(self):
        """
        Plot a diagram of the vertical temperature profiles of the Velis 65L
        
        """
        
        T_disp = np.flip(self.T_record1[-1])
        # print(T_disp)
 
        H = self.Height/2*100
        W = self.Diameter  *100

        ax = plt.figure(figsize=(4*1.5/1.5,3*1.5*2/1.5), constrained_layout=True)

        ax = plt.axes(xlim = (-W/2 - 0.1*W, W/2 + 0.1*W),ylim = (-H/2 - 0.1*H, H/2 + 0.1*H))
        plt.rcParams.update({'font.size':'17'})
        params = {
                  "text.usetex" : True,
                  "font.family" : "cm"}
        plt.rcParams.update(params)
        # contour
        xcont = np.array([-W/2,-W/2, W/2, W/2,-W/2])
        ycont = np.array([-H/2,H/2,H/2,-H/2,-H/2])
        plt.plot(xcont, ycont,'k', linewidth = 2.5)
        
        # layers
        T_disp_x = []
        T_disp_y = []
        for k in range(len(T_disp)):
            
            H_step = H/len(T_disp)
            x_sep = np.array([-W/2, W/2])
            y_sep = np.array([-H/2 + (k+1)*H_step, -H/2 + (k+1)*H_step])
            x_rect = np.array([-W/2, -W/2 , W/2, W/2,- W/2])
            y_rect = np.array([-H/2+ k*H_step, -H/2 + (1+k)*H_step, -H/2 + (1+k)*H_step,  -H/2+ k*H_step, -H/2+ k*H_step])
            plt.plot(x_sep, y_sep,'k', linewidth = 0.25)
            
            
            #color range
            T_min = 283.15
            T_max = 353.15+10
            
            T_disp_x.append((T_disp[k] - T_min)/(T_max - T_min)*W - W/2)
            T_disp_y.append(-H/2 + H_step/2 + k*H_step)
            
            
            R = (T_disp[k] - T_min)/(T_max - T_min)
            G = 0.2
            B = (T_disp[k] - T_max)/(T_min - T_max)
            plt.fill(x_rect, y_rect, color = (R,G,B))
        plt.plot(T_disp_x, T_disp_y,'k', linewidth = 2)
        
        # Plot x axis
        axis_xx = [-W/2, W/2]
        axis_xy = [-H/2 - 0.05*H/2, -H/2 - 0.05*H/2]
        
        plt.plot(axis_xx, axis_xy,'k', linewidth = 1.5)
        
        n_disp = 4
        size_line = 0.01
        step = W/n_disp
        
        T = np.linspace(T_min - 273.15, T_max - 273.15, n_disp+1)
        for i in range(n_disp+1):
            line_x = [-W/2 + i*step, - W/2 + i*step]
            line_y = [-H/2 - 0.05*H/2 -  H* size_line , - H/2 - 0.05*H/2 + H* size_line]
            
            plt.plot(line_x, line_y,'k', linewidth = 1.5)
            plt.text(-W/2 -1.5 + i*step, -H/2 - 0.05*H/2 - H*(size_line+0.05),''.join([ str(int(T[i])) ,'$^\circ$C']))
        
        ax.set_aspect('equal', 'box')
        plt.axis('off')
        
        # Resistor 
        exc = 0.08*W
        h_init = 0.1*H*0
        h_step = 0.18*W
        shift_res = 0.02*H
        x_res1  = np.array([0, 0, -exc, + exc, - exc, +exc])
        y_res1 = np.array([0, h_init, h_init + h_step/2, h_init + h_step/2 + h_step, h_init + h_step/2 + 2*h_step , h_init + 3*h_step+ h_step/2])
        
        x_res2  = np.flip(x_res1 + shift_res)
        y_res2  = np.flip(y_res1)
        x_res = np.concatenate([x_res1, x_res2])
        y_res = np.concatenate([y_res1, y_res2])
        color = 'darkslategrey'
        plt.plot(x_res - shift_res/2, y_res - H/2, color = color, linewidth = 2.5)
        plt.plot(x_res - shift_res/2 + 1.2*W, y_res - H/2, color = color, linewidth = 2.5)
        
    def plotConso(self):
        
        xdim = 5.5+1
        ydim = 4.3
        labelsize = 18
        ## General plot
        plt.figure(figsize=(xdim,ydim),constrained_layout=True)
        plt.rcParams.update({'font.size':16})
        params = {
                  "text.usetex" : True,
                  "font.family" : "cm"}
        plt.rcParams.update(params)
    
        
        
        # Plot 1, mass flow rate and water temperature
        ax = plt.subplot(2,1,1)
        plt.grid()
        time_vect = self.time_vect.copy()
        del time_vect[-1]
        l0 = plt.plot(np.array(time_vect)/3600, np.array(self.m_dot_w), linewidth = 2, color=u'#1f77b4')
        plt.xlabel('Time [h]',fontsize=18,  fontname="Times New Roman")
        plt.ylabel('$\dot{m}_{w,out}$ [kg/s]',fontsize=labelsize, fontname="Times New Roman" ,color=u'#1f77b4')            
        plt.xlim(0, max(np.array(time_vect)/3600) +0.02)
        plt.ylim(-0.01, 0.12)
        plt.yticks([0, 0.05, 0.1])
        # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                   # mode="expand", borderaxespad=0, ncol=3, fontsize=14)
        ax2=ax.twinx()
        l1 =ax2.plot(np.array(time_vect)/3600, np.array(self.T_w_out) -273.15, linewidth = 2,color=u'#ff7f0e')           
        ax2.set_ylabel( '$T_{w,out}$ [$^\circ$C]',fontsize=labelsize,color=u'#ff7f0e')      
        ax2.set_ylim(0, 91) 
        ax2.set_yticks([10,30,50,70,90])
        ax2.grid()
        # Plot 2, power and SOC of the tank      
        ax3 = plt.subplot(2,1,2)

        plt.grid()
        if self.EWH == True:
            l3 = plt.plot(np.array(time_vect)/3600, np.array(self.W_dot_cons_1), linewidth = 2, color=u'#1f77b4')
        elif self.HPWH == True:
            l3 = plt.plot(np.array(time_vect)/3600, np.array(self.W_dot_cons_tot), linewidth = 2, color=u'#1f77b4')
            
        if self.double == True:
            l4 = plt.plot(np.array(time_vect)/3600, np.array(self.W_dot_cons_2), linewidth = 2, color=u'#2ca02c')
            lns = l3+l4
            labels = ['HR1', 'HR2']
            plt.legend(lns, labels,fontsize=12, bbox_to_anchor=(0.22, 0.82))
        plt.xlabel('Time [h]',fontsize=18,  fontname="Times New Roman")
        plt.ylabel('$\dot{W}_{cons,elec}$ [W]',fontsize=labelsize, fontname="Times New Roman", color='k') 

        

        
        
        plt.xlim(0, max(np.array(time_vect)/3600) +0.02)
        # plt.ylim(-50, 3001)
        # plt.yticks([0, 750,1500,2250,3000])
        ax4=ax3.twinx()
        SOC = self.SOC.copy()
        del SOC[-1]
        l4 =ax4.plot(np.array(time_vect)/3600, np.array(SOC)*100, linewidth = 2,color=u'#ff7f0e')    
        ax4.grid()
        ax4.set_ylabel( 'SOC [$\%$]',fontsize=labelsize,color=u'#ff7f0e')   
        ax4.set_ylim(-1, 101) 
        ax4.set_yticks([0, 25,50,75,100])
        
        
    def plotTPointsVelis(self, index = -1,  exp_data = None):
        """
        Plot a diagram of the vertical temperature profiles of the Velis 65L
        
        """
        
        T_disp1 = np.flip(self.T_record1[index])
        T_disp2 = np.flip(self.T_record2[index])
        # print(T_disp)
        H = self.Height/2*100
        W = self.Diameter  *100   
        ax = plt.figure(figsize=(4*1.5/1.5,3*1.5*2/1.5), constrained_layout=True)
        
        scale = 0.1
        ax = plt.axes(xlim = (-W/2 - scale*W, W/2 + scale*W + 1.2*W), ylim = (-H/2 - scale*H, H/2 + scale*H))
        plt.rcParams.update({'font.size':'18'})
        params = {
                  "text.usetex" : True,
                  "font.family" : "cm"}
        plt.rcParams.update(params)
        # contour
        xcont = np.array([-W/2,-W/2, W/2, W/2,-W/2])
        ycont = np.array([-H/2,H/2,H/2,-H/2,-H/2])
        plt.plot(xcont, ycont,'k', linewidth = 2.5)
        
        # layers
        T_disp_x2 = []
        T_disp_y = []
        for k in range(len(T_disp1)):
            
            H_step = H/len(T_disp1)
            x_sep = np.array([-W/2, W/2])
            y_sep = np.array([-H/2 + (k+1)*H_step, -H/2 + (k+1)*H_step])
            x_rect = np.array([-W/2, -W/2 , W/2, W/2,- W/2])
            y_rect = np.array([-H/2+ k*H_step, -H/2 + (1+k)*H_step, -H/2 + (1+k)*H_step,  -H/2+ k*H_step, -H/2+ k*H_step])
            plt.plot(x_sep, y_sep,'k', linewidth = 0.25)
            
            
            #color range
            T_min = 283.15
            T_max = 353.15+10
            
            T_disp_x2.append((T_disp2[k] - T_min)/(T_max - T_min)*W - W/2)
            T_disp_y.append(-H/2 + H_step/2 + k*H_step)
            
            
            R = (T_disp2[k] - T_min)/(T_max - T_min)
            G = 0.2
            B = (T_disp2[k] - T_max)/(T_min - T_max)
            plt.fill(x_rect, y_rect, color = (R,G,B))
        plt.plot(T_disp_x2, T_disp_y,'k', linewidth = 2,zorder=1)
        
        # Plot x axis
        axis_xx = [-W/2, W/2]
        axis_xy = [-H/2 - 0.05*H/2, -H/2 - 0.05*H/2]
        
        plt.plot(axis_xx, axis_xy,'k', linewidth = 1.5)
        
        n_disp = 4
        size_line = 0.01
        step = W/n_disp
        
        T = np.linspace(T_min - 273.15, T_max - 273.15, n_disp+1)
        for i in range(n_disp+1):
            line_x = [-W/2 + i*step, - W/2 + i*step]
            line_y = [-H/2 - 0.05*H/2 -  H* size_line , - H/2 - 0.05*H/2 + H* size_line]
            
            plt.plot(line_x, line_y,'k', linewidth = 1.5)
            plt.text(-W/2-0.08*W + i*step*0.98, -H/2 - 0.05*H/2 - H*(size_line+0.05),''.join([ str(int(T[i])) ,'$^\circ$C']), fontsize = 12)
        
        "Second tank"
        gap = 0.2
        Shift = (1 + gap)*W
        # contour
        xcont = np.array([-W/2,-W/2, W/2, W/2,-W/2]) + Shift
        ycont = np.array([-H/2,H/2,H/2,-H/2,-H/2])
        plt.plot(xcont, ycont,'k', linewidth = 2.5)
        
        # layers
        T_disp_x1 = []
        T_disp_y = []
        for k in range(len(T_disp1)):
            
            H_step = H/len(T_disp1)
            x_sep = np.array([-W/2, W/2])+Shift
            y_sep = np.array([-H/2 + (k+1)*H_step, -H/2 + (k+1)*H_step])
            x_rect = np.array([-W/2, -W/2 , W/2, W/2,- W/2])+Shift
            y_rect = np.array([-H/2+ k*H_step, -H/2 + (1+k)*H_step, -H/2 + (1+k)*H_step,  -H/2+ k*H_step, -H/2+ k*H_step])
            plt.plot(x_sep, y_sep,'k', linewidth = 0.25)
            
            
            #color range
            T_min = 283.15
            T_max = 353.15+10
            
            T_disp_x1.append((T_disp1[k] - T_min)/(T_max - T_min)*W - W/2)
            T_disp_y.append(-H/2 + H_step/2 + k*H_step)
            
            
            R = (T_disp1[k] - T_min)/(T_max - T_min)
            G = 0.2
            B = (T_disp1[k] - T_max)/(T_min - T_max)
            plt.fill(x_rect, y_rect, color = (R,G,B))
        plt.plot(np.array(T_disp_x1) + Shift, T_disp_y,'k', linewidth = 2,zorder=1)
        
        # Plot x axis
        axis_xx = np.array([-W/2, W/2]) + Shift
        axis_xy = [-H/2 - 0.05*H/2, -H/2 - 0.05*H/2]
        
        plt.plot(axis_xx, axis_xy,'k', linewidth = 1.5)
        
        n_disp = 4
        size_line = 0.01
        step = W/n_disp
        
        T = np.linspace(T_min - 273.15, T_max - 273.15, n_disp+1)
        for i in range(n_disp+1):
            line_x = np.array([-W/2 + i*step, - W/2 + i*step])+Shift
            line_y = [-H/2 - 0.05*H/2 -  H* size_line , - H/2 - 0.05*H/2 + H* size_line]
            
            plt.plot(line_x, line_y,'k', linewidth = 1.5)
            plt.text(-W/2-0.05*W + i*step*0.98 + Shift, -H/2 - 0.05*H/2 - H*(size_line+0.05),''.join([ str(int(T[i])) ,'$^\circ$C']), fontsize = 12)
            
            
        ## Display of the connection between the tanks + heating resistor 
        color = 'darkslategrey'
        entry = 0.05
        diam = 0.06
        h_dist = 0.025
        x_p1 = [-W/2 + W - entry*W,  -W/2 + W + gap*W/2 - diam*W/2,  W/2 + gap*W/2 - diam*W/2 , W/2 + gap*W + entry*W]
        y_p1 = [-H/2 + h_dist*H + diam*W/2, -H/2 + h_dist*H + diam*W/2, H/2  - h_dist*H + diam*W/2, H/2  - h_dist*H + diam*W/2]
        
        x_p2 = [-W/2 + W - entry*W,  -W/2 + W + gap*W/2 + diam*W/2,  W/2 + gap*W/2 + diam*W/2 , W/2 + gap*W + entry*W]
        y_p2 = [-H/2 + h_dist*H - diam*W/2, -H/2 + h_dist*H - diam*W/2, H/2  - h_dist*H - diam*W/2, H/2  - h_dist*H - diam*W/2]
        
        plt.plot(x_p1, y_p1, color = color, linewidth = 2.5)
        plt.plot(x_p2, y_p2, color = color, linewidth = 2.5)
        
        # Arrow 
        len_arrow = 0.05
        small_shift = 0.0000*W
        plt.plot([W/2 + gap*W/2+small_shift, W/2 + gap*W/2+small_shift], [len_arrow *H, -len_arrow *H, ], color = color, linewidth = 2)
        plt.plot(W/2 + gap*W/2,-len_arrow *H, marker=(3, 0, 180), color = color, markersize = 6)
        
        
        # Resistor 
        exc = 0.08*W
        h_init = 0.1*H*0
        h_step = 0.18*W
        shift_res = 0.02*H
        x_res1  = np.array([0, 0, -exc, + exc, - exc, +exc])
        y_res1 = np.array([0, h_init, h_init + h_step/2, h_init + h_step/2 + h_step, h_init + h_step/2 + 2*h_step , h_init + 3*h_step+ h_step/2])
        
        x_res2  = np.flip(x_res1 + shift_res)
        y_res2  = np.flip(y_res1)
        x_res = np.concatenate([x_res1, x_res2])
        y_res = np.concatenate([y_res1, y_res2])
        
        plt.plot(x_res - shift_res/2, y_res - H/2, color = color, linewidth = 2.5)
        plt.plot(x_res - shift_res/2 + 1.2*W, y_res - H/2, color = color, linewidth = 2.5)
        
        
        # Plot text 
        plt.text(0 - 0.25*W, H/2 + 0.02*H, 'Second tank', fontsize = 12)
        plt.text(1*W, H/2 + 0.02*H, 'First tank', fontsize = 12)
        
        ax.set_aspect('equal', 'box')
        plt.axis('off')
        
        # When experimental data need to be plotted
        if exp_data != None:
            
            h_exp = exp_data['h'] 
            T_profile_1 = exp_data['T1'] 
            T_profile_2 = exp_data['T2'] 
            T_disp_x_1_exp = (T_profile_1 + 273.15 - T_min)/(T_max - T_min)*W - W/2 + Shift
            T_disp_x_2_exp = (T_profile_2 + 273.15 - T_min)/(T_max - T_min)*W - W/2 
            T_disp_y_exp = -H/2 + h_exp/2 
            plt.scatter(T_disp_x_1_exp, T_disp_y_exp,  marker="o", s=32,  facecolors='y', edgecolors='k',zorder=2)
            plt.scatter(T_disp_x_2_exp, T_disp_y_exp,  marker="o", s=32,  facecolors='y', edgecolors='k',zorder=2)
            

            ## Error calculation 
            
            h_vect_interp = np.linspace(min(T_disp_y),max(T_disp_y),100, endpoint=True)
            T1_interp = interp1d(T_disp_y, T_disp1, kind='cubic')
            T2_interp = interp1d(T_disp_y, T_disp2, kind='cubic')
            T1_interp_vect = T1_interp(h_vect_interp)
            T2_interp_vect = T2_interp(h_vect_interp)
            
            index_vect = np.zeros(len(T_disp_y_exp ))
            error_T1 = np.zeros(len(T_disp_y_exp ))
            error_T2 = np.zeros(len(T_disp_y_exp ))
            for k in range(len(T_disp_y_exp)):
                
                index_vect[k] = np.argmin(abs(h_vect_interp - T_disp_y_exp[k]))

                error_T1[k] = T1_interp_vect[int(index_vect[k])] - T_profile_1[k] - 273.15
                error_T2[k] = T2_interp_vect[int(index_vect[k])] - T_profile_2[k] - 273.15
            
            error_T = np.concatenate([error_T1, error_T2])
            MAE = sum(abs(error_T))/len(error_T) # Mean absolute error
            RMSE = np.sqrt(sum(error_T**2)/len(error_T))# Root mean squared error 
            print(f"Mean absolute error : {MAE:.1f} K")


    def plotTPointsNUOS(self, index = -1,  exp_data = None):
        """
        Plot a diagram of the vertical temperature profiles of the Velis 65L
        
        """

        T_disp = np.flip(self.T_record1[index])
        # print(T_disp)
     
        H = self.Height*100
        W = self.Diameter*150
    
        ax = plt.figure(figsize=(4*1.5/1.5,3*1.5*2/1.5), constrained_layout=True)
    
        ax = plt.axes(xlim = (-W/2 - 0.1*W, W/2 + 0.1*W),ylim = (-H/2 - 0.1*H, H/2 + 0.1*H))
        plt.rcParams.update({'font.size':'17'})
        params = {
                  "text.usetex" : True,
                  "font.family" : "cm"}
        plt.rcParams.update(params)
        # contour
        xcont = np.array([-W/2,-W/2, W/2, W/2,-W/2])
        ycont = np.array([-H/2,H/2,H/2,-H/2,-H/2])
        plt.plot(xcont, ycont,'k', linewidth = 2.5)
        
        # layers
        T_disp_x = []
        T_disp_y = []
        for k in range(len(T_disp)):
            
            H_step = H/len(T_disp)
            x_sep = np.array([-W/2, W/2])
            y_sep = np.array([-H/2 + (k+1)*H_step, -H/2 + (k+1)*H_step])
            x_rect = np.array([-W/2, -W/2 , W/2, W/2,- W/2])
            y_rect = np.array([-H/2+ k*H_step, -H/2 + (1+k)*H_step, -H/2 + (1+k)*H_step,  -H/2+ k*H_step, -H/2+ k*H_step])
            plt.plot(x_sep, y_sep,'k', linewidth = 0.25)
            
            
            #color range
            T_min = 283.15
            T_max = 323.15+10
            
            T_disp_x.append((T_disp[k] - T_min)/(T_max - T_min)*W - W/2)
            T_disp_y.append(-H/2 + H_step/2 + k*H_step)
            
            
            R = (T_disp[k] - T_min)/(T_max - T_min)
            G = 0.2
            B = (T_disp[k] - T_max)/(T_min - T_max)
            plt.fill(x_rect, y_rect, color = (R,G,B))
        plt.plot(T_disp_x, T_disp_y,'k', linewidth = 2)
        
        # Plot x axis
        axis_xx = [-W/2, W/2]
        axis_xy = [-H/2 - 0.05*H/2, -H/2 - 0.05*H/2]
        
        plt.plot(axis_xx, axis_xy,'k', linewidth = 1.5)
        
        n_disp = 5
        size_line = 0.01
        step = W/n_disp
        
        T = np.linspace(T_min - 273.15, T_max - 273.15, n_disp+1)
        for i in range(n_disp+1):
            line_x = [-W/2 + i*step, - W/2 + i*step]
            line_y = [-H/2 - 0.05*H/2 -  H* size_line , - H/2 - 0.05*H/2 + H* size_line]
            
            plt.plot(line_x, line_y,'k', linewidth = 1.5)
            plt.text(-W/2 -1.5 + i*step, -H/2 - 0.05*H/2 - H*(size_line+0.05),''.join([ str(int(T[i])) ,'$^\circ$C']))
        
        ax.set_aspect('equal', 'box')
        plt.axis('off')
        
        # Resistor 
        # exc = 0.08*W
        # h_init = 0.1*H*0
        # h_step = 0.18*W
        # shift_res = 0.02*H
        # x_res1  = np.array([0, 0, -exc, + exc, - exc, +exc])
        # y_res1 = np.array([0, h_init, h_init + h_step/2, h_init + h_step/2 + h_step, h_init + h_step/2 + 2*h_step , h_init + 3*h_step+ h_step/2])
        
        # x_res2  = np.flip(x_res1 + shift_res)
        # y_res2  = np.flip(y_res1)
        # x_res = np.concatenate([x_res1, x_res2])
        # y_res = np.concatenate([y_res1, y_res2])
        # color = 'darkslategrey'
        # plt.plot(x_res - shift_res/2, y_res - H/2, color = color, linewidth = 2.5)
        # plt.plot(x_res - shift_res/2 + 1.2*W, y_res - H/2, color = color, linewidth = 2.5)
        
        # When experimental data need to be plotted
        if exp_data != None:
            
            h_exp = exp_data['h'] 
            T_profile = exp_data['T'] 
            T_disp_x_1_exp = (T_profile + 273.15 - T_min)/(T_max - T_min)*W - W/2 

            T_disp_y_exp = -H/2 + h_exp
            plt.scatter(T_disp_x_1_exp, T_disp_y_exp,  marker="o", s=32*2,  facecolors='y', edgecolors='k',zorder=2)

            ## Error calculation 
            
            h_vect_interp = np.linspace(min(T_disp_y),max(T_disp_y),100, endpoint=True)
            T1_interp = interp1d(T_disp_y, T_disp, kind='cubic')
            T1_interp_vect = T1_interp(h_vect_interp)

            
            index_vect = np.zeros(len(T_disp_y_exp ))
            error_T = np.zeros(len(T_disp_y_exp ))
            for k in range(len(T_disp_y_exp)):
                
                index_vect[k] = np.argmin(abs(h_vect_interp - T_disp_y_exp[k]))

                error_T[k] = T1_interp_vect[int(index_vect[k])] - T_profile[k] - 273.15
            
            MAE = sum(abs(error_T))/len(error_T) # Mean absolute error
            RMSE = np.sqrt(sum(error_T**2)/len(error_T))# Root mean squared error 
            print(f"Mean absolute error : {MAE:.1f} K")       
        

def save_results(name, results):
    # 'list_file' specifies the precise file directory
    name = file_name(name)
    list_file = ['..\\data\\Simulations\\' , name ,'.pkl']
    
    # The complete file name is built up
    filename = "".join(list_file)       
    
    # 'file' points at the file in which the pickled object will be written
    with open(filename, "wb") as file:
        # The dump() method of the pickle module in Python, converts a Python object hierarchy into a byte stream. 
        # This process is also called as serilaization.
        pickle.dump(results, file)


def open_results(name, path = None):
    # 'list_file' specifies the precise file directory
    if path == None:
        path = '..\\data\\Simulations\\'
    list_file = [path , name ,'.pkl']
    # The complete file name is built up
    filename = "".join(list_file)       
    # 'file' points at the file to be opened
    with open(filename, "rb") as file:
        # The load() method of Python pickle module reads the pickled byte stream of one or more python objects 
        # from a file object
        output = pickle.load(file)
    return output
    

def file_name(filename):
    list_path = ['..\\data\\Simulations\\' , filename ,'.pkl']
    filename_new = filename
    path = "".join(list_path) 
    i = 2
    while os.path.isfile(path):
        filename_list = [ filename , '_' ,str(i)]
        filename_new = "".join(filename_list) 
        list_path = ['..\\data\\simulations\\' , filename_new ,'.pkl']
        path = "".join(list_path) 
        i += 1
    return filename_new         