#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, stderr, exit

if len(argv) < 3:
    print("{}: [data file] [algorithm name]", argv[0], file=stderr)
    exit(1)

narrow_exper = pd.read_csv(argv[1], sep='\t', comment='#', float_precision="high", header=None, names=["m", "k", "l", "n", argv[2], "Goto", "error"])
narrow_exper["Narrowed Dim."] = narrow_exper[["m", "k", "l", "n"]].idxmin(axis=1)
narrow_exper["N"] = narrow_exper.apply(lambda row: row["k"] if row["Narrowed Dim."] == "m" else row["m"], axis=1)
narrow_exper2 = narrow_exper.copy()
narrow_exper2.set_index(["Narrowed Dim.", "N"], inplace=True)
narrow_exper2.drop(["m", "n", "k", "l", "error"], axis=1, inplace=True)

fig = plt.figure()
i = 1
for name, group in narrow_exper2.groupby(level="Narrowed Dim."):
    ax = fig.add_subplot(220 + i)
    group2 = group.copy()
    group2.index = group2.index.droplevel()
    group2.plot(ax=ax, title="Narrow {}".format(name))
    ax.set_xlabel("Large dimensions")
    ax.set_ylabel("GFlops/s")
    i = i + 1
plt.show()
