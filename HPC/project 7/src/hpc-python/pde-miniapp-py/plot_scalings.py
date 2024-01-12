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
dfp = pd.read_csv('strong_scaling_python.csv')
# Group the data by grid size
grouped = df.groupby('grid_size')
groupedp = dfp.groupby('grid_size')

plt.figure(figsize=(10, 6))
# Plot each group
for name, group in grouped:
    plt.plot(group['ranks'], group['iters_per_second'], label=f"C++ grid {name}")

for name, group in groupedp:
    plt.plot(group['ranks'], group['iters_per_second'], linestyle='--', label=f"Python grid {name}")

# Add labels and title
plt.xlabel("Number of Processes")
plt.ylabel("Iterations Per Second")
plt.title("Iterations Per Second vs Number of Processes")
plt.suptitle('C++ vs Python, pde-miniapp strong scaling')
plt.grid()
plt.xticks(df['ranks'].unique())

result_df = pd.concat([df, dfp], axis=0)
result_df = result_df.reset_index(drop=True)

plt.yticks(select_yticks(result_df,1300),fontsize=8)

# Add a legend
plt.legend()

# Show the plot
#plt.show()
plt.savefig('strong_scaling_python')
# Correcting the positioning of the x-tick labels

# Assuming the file is named 'data.csv' and is located in the current directory

# Read the CSV file into a DataFrame
df2 = pd.read_csv('weak_scaling.csv')
df2p = pd.read_csv('weak_scaling_python.csv')

# Create a new column for x-axis labels (ranks and grid size)
df2['x_label'] = df2.apply(lambda row: f"{row['ranks']}, {row['grid_size']}", axis=1)
df2p['x_label'] = df2p.apply(lambda row: f"{row['ranks']}, {row['grid_size']}", axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df2['x_label'], df2['iters_per_second'],label="C++")
plt.plot(df2p['x_label'], df2p['iters_per_second'],label="Python")

# Adding labels and title
plt.xlabel("Ranks, Grid Size")
plt.ylabel("Iterations per Second")
plt.title("Iterations per Second vs Ranks and Grid Size")
plt.suptitle('C++ pde-miniapp weak scaling')
# Correcting x-tick labels to appear directly under each plot point
# The labels show ranks and grid size on separate lines
plt.grid()
result2_df = pd.concat([df2, df2p], axis=0)
result2_df = result2_df.reset_index(drop=True)
plt.yticks(select_yticks(result2_df,10),fontsize=7)
plt.xticks(df2.index, [f"{r}\n{g}" for r, g in zip(df2['ranks'], df2['grid_size'])])

plt.legend()

# Show plot with adjustments
plt.tight_layout()  # Adjust layout for better fit
#plt.show()
plt.savefig('weak_scaling_python')


