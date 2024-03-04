# The Water Heaters library

### Description
This Python library allows to model electrical or heat pump water heaters, including effects such as stratification, ambient heat losses, convection and advection. It can be used for the simulation of individual units, or to create a pool of water heaters used as a Virtual Power Plant (VPP) providing services to the grid.
 
### Features
The water heater model is a linear model with the following features.
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


