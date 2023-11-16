#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.axis import Axis

# Read the CSV file
file_path = 'strong_scalling.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extracting data from the dataframe
n_values = data['n'].unique()
p_values = data['p']
time_values = data['time']
formatter = ScalarFormatter()
formatter.set_scientific(False)


# Plotting the data
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Strong Scalling')
ax.plot(p_values, time_values, marker='o', linestyle='-', color='b')
ax.set_title(f'Time vs. Number of Processes\nFixed workload per processor')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
plt.minorticks_off()
ax.set_xticks(p_values)
ax.set_yticks(time_values)
ax.yaxis.set_major_formatter(formatter)
ax.set_yticklabels([f"{t:.2f}" for t in time_values])
fig.subplots_adjust(bottom=0.15)
ax.grid(True)
fig.savefig('strong_scalling')

'''
# Plotting the data
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Strong Scalling')
ax.plot(p_values, time_values, marker='o', linestyle='-', color='b')
ax.set_title(f'Time vs. Number of Processes, w/ fixed {n_values} matrix size')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_yticks(time_values)
'''

# p against T1/P*Tp
plt.clf()

efficiency = []
for p,t in zip(p_values,time_values):
    efficiency.append(time_values[0]/(p*t))

fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Strong Scalling')
ax.plot(p_values, efficiency, marker='o', linestyle='-', color='b')
ax.set_title(f'Time vs. Number of Processes\nFixed workload per processor')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Efficiency factor')
ax.set_yscale('log')
plt.minorticks_off()
ax.set_xticks(p_values)
ax.set_yticks(efficiency)
ax.yaxis.set_major_formatter(formatter)
ax.set_yticklabels([f"{t:.2f}" for t in efficiency], fontsize=8)
fig.subplots_adjust(bottom=0.15)
ax.grid(True)
fig.savefig('strong_scalling_efficiency')