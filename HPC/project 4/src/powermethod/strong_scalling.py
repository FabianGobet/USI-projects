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
ax.set_title('Time vs. Number of Processes, w/ fixed overall workload')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Time (seconds)')
ax.set_yscale('log')
#ax.set_yticks(time_values)
ax.set_xticks(p_values)
Axis.set_major_formatter(ax.yaxis,formatter)
ax.set_yticklabels(time_values)
ax.grid(True)
fig.savefig('strong_scalling')
