#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

def select_yticks(df, alfa):
    # Sort the values and remove duplicates
    sorted_values = sorted(set(df['iters_per_second']))

    # Initialize the result list with the first value
    yticks = [sorted_values[0]]

    # Iterate through the sorted list
    for value in sorted_values[1:]:
        # Check if the current value is at least 'alfa' distance from the last added value
        if value - yticks[-1] >= alfa:
            yticks.append(value)

    return yticks

# Read the CSV file
df = pd.read_csv('strong_scaling.csv')

# Group the data by grid size
grouped = df.groupby('grid_size')

plt.figure(figsize=(10, 6))
# Plot each group
for name, group in grouped:
    plt.plot(group['ranks'], group['iters_per_second'], label=f"C++ grid {name}")

# Add labels and title
plt.xlabel("Number of Processes")
plt.ylabel("Iterations Per Second")
plt.title("Iterations Per Second vs Number of Processes")
plt.suptitle('C++ pde-miniapp strong scaling')
plt.grid()
plt.xticks(df['ranks'].unique())
plt.yticks(select_yticks(df,1300),fontsize=8)

# Add a legend
plt.legend()

# Show the plot
#plt.show()
#plt.savefig('strong_scaling')
# Correcting the positioning of the x-tick labels

# Assuming the file is named 'data.csv' and is located in the current directory
file_path = 'weak_scaling.csv'

# Read the CSV file into a DataFrame
df2 = pd.read_csv(file_path)

# Create a new column for x-axis labels (ranks and grid size)
df2['x_label'] = df2.apply(lambda row: f"{row['ranks']}, {row['grid_size']}", axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df2['x_label'], df2['iters_per_second'], marker='o')

# Adding labels and title
plt.xlabel("Ranks, Grid Size")
plt.ylabel("Iterations per Second")
plt.title("Iterations per Second vs Ranks and Grid Size")
plt.suptitle('C++ pde-miniapp weak scaling')
# Correcting x-tick labels to appear directly under each plot point
# The labels show ranks and grid size on separate lines
plt.grid()
plt.xticks(df2.index, [f"{r}\n{g}" for r, g in zip(df2['ranks'], df2['grid_size'])])
plt.yticks(df2['iters_per_second'])

# Show plot with adjustments
plt.tight_layout()  # Adjust layout for better fit
#plt.show()
#plt.savefig('weak_scaling')


strong_eff = {}

for g in df['grid_size'].unique():
    dtemp = df[df['grid_size'] == g ]
    t0 = dtemp['iters_per_second'].iloc[0]
    strong_eff.update({g:[ t0/(p*t)for p,t in zip(dtemp['ranks'],dtemp['iters_per_second'])]})


plt.figure(figsize=(10, 6))
# Plot each group
for g,ys in strong_eff.items():
    plt.plot(df['ranks'].unique(),ys,label=f"C++ grid {g}")

# Add labels and title
plt.xlabel("Number of Processes")
plt.ylabel("% efficiency")
plt.title("Efficiency vs Number of Processes")
plt.suptitle('C++ pde-miniapp strong scaling efficiency')
plt.grid()
plt.yscale('log')
plt.xticks(df['ranks'].unique())
#plt.yticks(select_yticks(df,1300),fontsize=8)

# Add a legend
plt.legend()

# Show the plot
plt.show()
#plt.savefig('strong_scaling_efficiency')