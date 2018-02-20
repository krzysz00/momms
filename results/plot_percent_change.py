#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sys import argv, stderr, exit

if len(argv) < 3:
    print("{}: [data file] [plot title] [[xlabel]] [[file]]".format(argv[0]), file=stderr)
    exit(1)

exper = pd.read_csv(argv[1], sep=' ', comment='#', float_precision="high", header=None, index_col=False, names=["N", "Percent change"])
to_plot = exper.copy()
to_plot["Percent change"] = to_plot["Percent change"] * 100
to_plot["N"] = to_plot["N"].apply(lambda x: str(x) if x % 256 == 0 else '')
ax = to_plot.plot(x="N", y="Percent change", title=argv[2], xlim=(0, exper.index[-1]), ylim=(-15, 15),
                  kind='bar', legend=None)
ax.set_xlabel("N" if len(argv) == 4 else argv[3])
ax.set_ylabel("Percent change")

if len(argv) < 5:
    plt.show()
else:
    plt.savefig(argv[4])
