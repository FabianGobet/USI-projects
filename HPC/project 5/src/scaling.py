#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.axis import Axis

# Read the CSV file
file_path = 'scaling.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extracting data from the dataframe
x_values = data['grid'].unique()
threads = data['threads'].unique()
ranks = data['ranks'].unique()
y_list_values = {}

for r in ranks:
    for t in threads:
        key = f"r:{r},t:{t}"
        value = data.loc[(data['ranks'] == r) & (data['threads'] == t)].sort_values('time')['time']
        y_list_values.update({key: value})
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
'''
for r in ranks:
    fig, ax = plt.subplots()
    fig.suptitle("Scaling with "+str(r)+" ranks.")
    for k,v in y_list_values.items():
        if(int(r)==int(k.split(",")[0].split(":")[1])):
            col = colors[int(k.split(":")[2])-1]
            ax.plot(x_values, v, marker='o', color = col, linestyle='-', label=k)  
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    ax.set_xlabel('grid size')
    ax.set_ylabel("time (s)")
    ax.set_xticks(x_values)
    plt.show()
    plt.clf()
'''

d1024 = data.loc[(data['grid']==1024)].sort_values('time')[0:2]
print(d1024)

d512 = data.loc[(data['grid']==512)].sort_values('time')[0:2]
print(d512)

d256 = data.loc[(data['grid']==256)].sort_values('time')[0:2]
print(d256)

d128 = data.loc[(data['grid']==128)].sort_values('time')[0:2]
print(d128)