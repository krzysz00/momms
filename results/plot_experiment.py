#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, stderr, exit
from math import ceil

if len(argv) < 4:
    print("{}: [data file] [algorithm name] [plot title] [[xlabel]] [[file]]".format(argv[0]), file=stderr)
    exit(1)

exper = pd.read_csv(argv[1], sep='\t', comment='#', float_precision="high", header=None, names=["m", "n", "k", "l", argv[2], "MOMMS BLIS algo.", "error"])
to_plot = exper[["m", argv[2], "MOMMS BLIS algo."]].copy()
to_plot.set_index("m", inplace=True)
x_max = int(ceil(to_plot.index[-1] / 1000.0)) * 1000
ax = to_plot.plot(title=argv[3], xlim=(0, x_max), ylim=(0, 50),
                  style='.')
ax.set_xlabel("N" if len(argv) == 4 else argv[4])
ax.set_ylabel("GFlops/s")
if len(argv) < 6:
    plt.show()
else:
    plt.savefig(argv[5])
