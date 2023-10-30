#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

df = pd.read_csv("mydata.csv")
mesh_initials = df['i'].unique()

for i in mesh_initials:
    plt.clf()
    thread_initials = df[df['i'] == i]['j'].unique()
    ytics = np.array([])
    for j in thread_initials:
        stt = df.loc[(df['i'] == i) & (df['j'] == j), ['threads', 'size','time']]
        x = [n**2 for n in stt['size']]
        if len(x)!=1:
            y = stt['time']
            for tic in y:
                ytics = np.append(ytics, tic)
            lbl = "["+ ",".join(str(s) for s in stt['threads'])+ "] threads" 
            plt.plot(x,y,label=lbl)

    plt.xscale('log', base=2)
    plt.xticks([n**2 for n in df[df['i'] == i]['size'].unique()])
    plt.yticks(np.unique(ytics), fontsize=7)
    plt.title("Weak scalling\nInitial mesh {}x{}.".format(i,i))
    plt.xlabel("mesh size")
    plt.ylabel("seconds")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("weak{}.png".format(i), dpi=300)
