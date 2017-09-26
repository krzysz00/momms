#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, stderr, exit
from memory_usage import memory_mine, memory_goto

if len(argv) < 4:
    print("{}: [data file] [algorithm name] [plot title] [[xlabel]]",
          argv[0], file=stderr)
    exit(1)

exper = pd.read_csv(argv[1], sep='\t', comment='#', float_precision="high",
                    header=None,
                    names=["m", "n", "k", "l", "flopsa", "flopsb", "error"])

exper[argv[2]] = exper.apply(lambda r: memory_mine(r["m"], r["n"],
                                                   r["k"], r["l"]),
                             axis=1)
exper["Goto"] = exper.apply(lambda r: memory_goto(r["m"], r["n"],
                                                  r["k"], r["l"]),
                            axis=1)

to_plot = exper[["m", argv[2], "Goto"]].copy()
to_plot.set_index("m", inplace=True)
ax = to_plot.plot(title=argv[3], xlim=(0, to_plot.index[-1]))
ax.set_xlabel("N" if len(argv) == 4 else argv[4])
ax.set_ylabel("Additional allocations (MiB)")
plt.show()
