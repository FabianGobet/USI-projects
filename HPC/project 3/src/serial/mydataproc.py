#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("mydata.csv")
numt_values = np.sort(df['threads'].unique())
numt = np.max(numt_values)
mesh_values = np.sort(df['size'].unique())

for mesh in mesh_values:
    
    filtered_df = df[df['size'] == mesh]
    sorted_df = filtered_df.sort_values(by='threads')
    time_array = sorted_df['time'].to_numpy()
    numt = time_array.size

    plt.clf()
    plt.plot(np.linspace(1,numt,num=numt),time_array)
    plt.title("Simulation times for mesh size of {}x{} over {} threads".format(mesh,mesh,numt))
    plt.xlabel("no. of threads")
    plt.ylabel("seconds")
    plt.grid(True, which="both", ls="--")
    plt.xticks(numt_values)
    #plt.legend()
    plt.savefig("{}.png".format(mesh), dpi=300)
    #plt.show()

