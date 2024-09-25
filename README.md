# The Water Heaters library

### Description
This Python library allows to model electrical or heat pump water heaters, including effects such as stratification, ambient heat losses, convection and advection. It can be used for the simulation of individual units, or to create a pool of water heaters used as a Virtual Power Plant (VPP) providing services to the grid.
 
### Features
The water heater model is a linear model with the following features.
The model takes as first inputs the storage dimensions:
    - Volume or Diameter
    - Height
    - Presence of an Electrical Water Heater (EWH)
    - Presence of a Heat Pump Water Heater (HPWH)
    
The heating system is then defined:
For a EWH, the electrical resistance peak power consumption is required
    - Q_dot_peak_E
The heating system dimensions are then required:
    - z_init_E (starting height of the electrical resistance)
    - z_init_HP (starting height of the heat pump coil)
    - height_E (height of the electrical resistance)
    - height_Hp (height of the HP coil)
    
The model parameters need then to be entered: 
    - h_amb: heat exchange with the ambiance (from the water)
    - h_ref: heat exchange with the refrigerant (to the water)
    - delta: parameters mused to model the inversion of layer scheme (>>10000)
    - H_mix: height of mixing with the supply water at the bottom of the storage tank (+- 15cm)
    - eps_is: isentropic efficiency of the compressor (+- 0.65)
    - W_dot_el_basis: auxiliaries electricty consumption of the heat pump (100 W)
    
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


### Quick start

If you want to download the latest version from github for use or development purposes, make sure that you have git and an [anaconda](https://www.anaconda.com/distribution/) or [miniconda](https://docs.conda.io/projects/miniconda/) installed and type the following:

```bash
git clone https://github.com/NLeclercq27/Water-Heater.git
cd Water-Heater
conda env create  # Automatically creates environment based on environment.yml
conda activate waterheaters # Activate the environment
```

The above commands create a dedicated environment so that your conda configuration remains clean from the required dependencies installed.

To check that everything runs fine, you can run the example python files provided in the scripts folder.

### Digital Twin use (pool of water heaters)

A new script called digital_twin (in the script folder) allows the simulation of different water heaters in parallel. 
This aggregation of water heaters (also called pool of water heaters) constitutes a digital twin allowing to assess the electricity consumption of the pool under stochastic water consumption profiles. 

Four kinds of water heater can be used: 
- Random heat pump water heater (number defined with the variable N_random_HP): heat pump water heater of variable dimensions
- Random electrical water heater (number defined with the variable N_random_E): electrical water heater of variable dimensions
- VELIS water heater (number defined with the variable N_VELIS): VELIS water heater (model from Ariston)
- NUOS water heater (number defined with the variable N_NUOS): NUOS water heater (model from Ariston)

The  control of the pool over time can be done directly in the digital_twin script using the probe temperatures
Several control strategies are pre-defined:
- "tracking_SP": - The system stops heating when the set point is exceeded by a temperature difference DT_high
                 - The system starts heating when the control temperatrure is DT_low below the set point
                 - switch2 always has the priority on switch 1 for the velis
                 - The additional heating resistor is not used for the HPWH
- "mid_day_night": - The system starts heating at midday or midnight and stops when the temperature 
                  setpoint + DThigh is reached
- "full_heat": - heat the system until the maximum water temperature is reached (80°C in every water heater)
                  strategy for electricity storage (gives the maximum power and energy that can be stored over time)
               - heat the HPWH storage using the electrical resistor as well . The default is 'tracking_SP'.
The results are saved in two csv files:
- The file with the extension "charact" gives the parameters of each water heater of the pool.
- The file with the extension "time_series" gives the variables (electrical consumption, water consumption,...)
        of each water heater varying with time.
  
Please contact Nicolas Leclercq (N.Leclercq@uliege.be) for more information.
