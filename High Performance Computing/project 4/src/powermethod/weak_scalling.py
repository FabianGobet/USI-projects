#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.axis import Axis

# Read the CSV file
file_path = 'weak_scalling.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extracting data from the dataframe
n_values = data['n'].unique()
p_values = data['p']
time_values = data['time']
formatter = ScalarFormatter()
formatter.set_scientific(False)


# Plotting the data
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Weak Scalling')
ax.plot(p_values, time_values, marker='o', linestyle='-', color='b')
ax.set_title('Time vs. Number of Processors\nFixed workload per processor')
ax.set_xlabel('Number of Processors\nMatrix y-dim')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_xticks(p_values)
ax.set_yticks(time_values[1:])
Axis.set_major_formatter(ax.yaxis,formatter)
ax.set_xticklabels([str(l1)+"\n"+str(l2) for l1,l2 in zip(p_values,n_values)])
fig.subplots_adjust(bottom=0.15)
ax.grid(True)
fig.savefig('weak_scalling')

plt.clf()
efficiency = []
for p,t in zip(p_values,time_values):
    efficiency.append(time_values[0]/(p*t))


fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Weak Scalling efficiency')
ax.plot(p_values, efficiency, marker='o', linestyle='-', color='b')
ax.set_title('Time vs. Number of Processors\nFixed workload per processor')
ax.set_xlabel('Number of Processors\nMatrix y-dim')
ax.set_ylabel('Efficiency factor')
ax.set_yscale('log')
ax.set_xticks(p_values)
ax.set_yticks(efficiency)
Axis.set_major_formatter(ax.yaxis,formatter)
ax.set_xticklabels([str(l1)+"\n"+str(l2) for l1,l2 in zip(p_values,n_values)])
fig.subplots_adjust(bottom=0.15)
ax.grid(True)
fig.savefig('weak_scalling_efficiency')