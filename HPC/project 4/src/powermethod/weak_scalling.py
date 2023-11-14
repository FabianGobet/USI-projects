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
ax.set_title('Time vs. Number of Processors w/ fixed workload')
ax.set_xlabel('Number of Processors\nMatrix y-dim')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
ax.set_xticks(p_values)
ax.set_yticks(time_values[1:])
Axis.set_major_formatter(ax.yaxis,formatter)
ax.set_xticklabels([str(l1)+"\n"+str(l2) for l1,l2 in zip(p_values,n_values)])
fig.subplots_adjust(bottom=0.15)
ax.grid(True)

'''
plt.plot(p_values, time_values, marker='o', linestyle='-', color='b')
plt.title('Time vs. Number of Processes')
plt.xlabel('Number of Processors (p)')
plt.ylabel('Time (seconds)')
plt.xticks(p_values)
for tick, label in zip(p_values, n_values):
    plt.text(tick, -0.5, str(label), ha='center', va='center')
#plt.xticks(p_values,labels=[str(n) for n in n_values],minor=True)#[str(l1)+"\n"+str(l2) for l1,l2 in zip(p_values,n_values)])

plt.yscale('log')
plt.yticks(time_values[1:])
plt.grid(True)

plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
'''

fig.savefig('weak_scalling')
