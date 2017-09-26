#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, stderr, exit
from memory_usage import memory_mine, memory_goto

if len(argv) < 4:
    print("{}: [data file] [algorithm name] [plot title]".format(argv[0]), file=stderr)
    exit(1)

narrow_exper = pd.read_csv(argv[1], sep='\t', comment='#',
                           float_precision="high", header=None,
                           names=["m", "n", "k", "l", "flopsa", "flopsb", "error"])

narrow_exper["Narrowed Dim."] = narrow_exper[["m", "n", "k", "l"]].idxmin(axis=1)
narrow_exper["N"] = narrow_exper.apply(lambda row: row["k"]
                                       if row["Narrowed Dim."] == "m"
                                       else row["m"], axis=1)
narrow_exper[argv[2]] = narrow_exper.apply(lambda r:
                                           memory_mine(r["m"], r["n"],
                                                       r["k"], r["l"]),
                                           axis=1)
narrow_exper["Goto"] = narrow_exper.apply(lambda r:
                                          memory_goto(r["m"], r["n"],
                                                      r["k"], r["l"]),
                                          axis=1)

narrow_exper2 = narrow_exper.copy()
narrow_exper2.set_index(["Narrowed Dim.", "N"], inplace=True)
narrow_exper2.drop(["m", "n", "k", "l", "flopsa", "flopsb", "error"],
                   axis=1, inplace=True)

fig = plt.figure()
fig.suptitle(argv[3])
i = 1
for name, group in narrow_exper2.groupby(level="Narrowed Dim."):
    ax = fig.add_subplot(220 + i)
    group2 = group.copy()
    group2.index = group2.index.droplevel()
    group2.plot(ax=ax, title="Narrow {}".format(name),
                xlim=(0, group2.index[-1]))
    ax.set_xlabel("N (large dimensions)")
    ax.set_ylabel("Additional Allocations (MiB)")
    i = i + 1
plt.show()
