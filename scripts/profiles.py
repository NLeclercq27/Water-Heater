import os, sys
import pandas as pd
import time
import matplotlib.pyplot as plt
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
import source as procF
from datetime import datetime
# Initialize the simulation
obj = procF.dlt.WaterHeaterPool()
profiles = []
# for i in range(7):
#     data = obj.DHW_load_gen(nday =7)
#     profiles.append(data)
# profiles = pd.DataFrame(profiles).T.to_csv('DHW_load_gen3.csv')
data = obj.DHW_load_gen(nday = 365)
start_time =  '00:00:00'

num_rows = len(data)

time_index = pd.date_range(start=start_time, periods=num_rows, freq='min')               
df = pd.DataFrame(data, index=time_index)
df.index = df.index.strftime('%d-%m-%Y %I:%M:%S.%f %p') 
df.index = [date[:-3] + '000' for date in df.index]
df.index.name = 'Time'
df.columns = ['WaterConsumption']
df['WaterConsumption'] = df['WaterConsumption']*60

# df.plot()
df.to_csv('DHW_load_gen1.csv')
import plotly.express as px

fig = px.line(df, x=df.index, y='WaterConsumption', title='Load Over Time', labels={'x': 'Time', 'y': 'Load'})
# fig.show()