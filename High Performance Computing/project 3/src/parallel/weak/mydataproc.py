#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv("mydata.csv")
mesh_initials = df['i'].unique()

for i in mesh_initials:
    plt.clf()
    thread_initials = df[df['i'] == i]['j'].unique()
    pre_ytics = []
    for j in thread_initials:
        stt = df.loc[(df['i'] == i) & (df['j'] == j), ['threads', 'size','time']]
        x = [n for n in stt['size']]
        if len(x)!=1:
            y = stt['time']
            for tic in y:
                pre_ytics.append(tic)
            lbl = "["+ ",".join(str(s) for s in stt['threads'])+ "] threads" 
            plt.plot(x,y,label=lbl)

    """
    M = np.max(pre_ytics)
    m = np.min(pre_ytics)
    no_intervals = 20
    interval_size = (M-m)/no_intervals
    yticks = []
    lbound = m
    while lbound <= M:
        rbound = lbound + interval_size
        ticks = (pre_ytics >= lbound) & (pre_ytics<rbound)
        if len(ticks)>0:
            for t in ticks:
                yticks.append(t) 
        lbound = lbound + interval_size
    """
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    

    #plt.yticks(yticks, fontsize=7)
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks([n for n in df[df['i'] == i]['size'].unique()])
    plt.yticks(pre_ytics, fontsize=7)
    plt.title("Weak scalling\nInitial mesh {}x{}.".format(i,i))
    plt.xlabel("mesh side size (base 2 scale)")
    plt.ylabel("seconds (base 2 scale)")
    plt.grid(True, which="major", ls="--")
    plt.legend()
    plt.savefig("weak{}.png".format(i), dpi=300)
