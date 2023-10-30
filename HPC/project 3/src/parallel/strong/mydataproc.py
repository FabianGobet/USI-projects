#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("mydata.csv")
dt = pd.read_csv("mydata_serial.csv")
columns = df.columns[2:5:2]

numt_values = np.sort(df['threads'].unique())
numt = np.max(numt_values)
mesh_values = np.sort(df['size'].unique())
titles = ["Simulation times\n", "Iterations per second\n"]


for col,tit in zip(columns,titles):
    for mesh in mesh_values:

        plt.clf()
        serial_dt = dt[dt['size'] == mesh][col].to_numpy()
        values_array = df[df['size'] == mesh].sort_values(by='threads')[col].to_numpy()
        plt.plot(np.linspace(1,numt,num=2),np.tile(serial_dt.squeeze(),2), label="serial", color="blue")
        plt.plot(numt_values,values_array, label="threads", color="green")
        plt.title(tit+"{}x{} mesh, 1-{} threads and serial".format(mesh,mesh,numt))
        plt.xlabel("no. of threads")
        plt.ylabel("seconds" if col=='time' else "iterations/second")
        plt.grid(True, which="both", ls="--")
        plt.xticks(numt_values, fontsize=7)
        
        pre_yticks = np.concatenate((values_array.reshape(-1),serial_dt.reshape(-1)))
        if col == "iters_cg": print(pre_yticks)
        M = np.max(pre_yticks)
        m = np.min(pre_yticks)
        no_intervals = 20
        interval_size = (M-m)/no_intervals
        ticks_list = []
        lbound = m
        while lbound <= M:
            rbound = lbound + interval_size
            ticks = pre_yticks[(pre_yticks>=lbound) & (pre_yticks<rbound)]
            if len(ticks)>0:
                ticks_list.append(np.max(ticks))
            lbound = lbound + interval_size
        
        yticks = np.array(ticks_list)
        plt.yticks(yticks, fontsize=5)
        plt.legend()
        plt.savefig("{}{}.png".format(col,mesh), dpi=300)

