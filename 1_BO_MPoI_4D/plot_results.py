import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SMALL_SIZE = 25
plt.rc('font', family='serif')
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=SMALL_SIZE)
plt.rc('path', simplify=False)

# Exact Pareto curve
f_1_pareto = np.linspace(0.0, 1.0, 500)
f_2_pareto = np.sqrt(1.0 - (f_1_pareto**2.0))

# --- ONLY LOAD THIS ONE FILE ---
DATA_FILE = "DTLZ2_Function_MPoI_b143s43_r2023-12-30.npz"
initial_data = np.load(DATA_FILE)
objectives = initial_data["arr_1"]

# Single figure/axis instead of 5x5 grid
fig, ax0 = plt.subplots(figsize=(9, 8))

ax0.scatter(objectives[:43, 1], objectives[:43, 0],
            marker="x", color="blue", s=120)
ax0.plot(objectives[43:, 1], objectives[43:, 0],
         marker="o", ms=15, mec="green", mfc="None", ls="None")

ax0.minorticks_on()
ax0.grid(True, which="both", alpha=0.25)
ax0.set_ylim(-0.0, 2.5)
ax0.set_xlim(-0.0, 2.5)
ax0.hlines(0.9, -0.5, 2.6, colors="r", linestyles="--")
ax0.vlines(0.9, -0.5, 2.6, colors="r", linestyles="--")
ax0.xaxis.set_major_locator(MultipleLocator(0.5))

ax0.plot(f_1_pareto, f_2_pareto, "k-")
ax0.set_xlabel(r"$f_1 \ (-)$")
ax0.set_ylabel(r"$f_2 \ (-)$")

plt.show()
fig.savefig("BO_MPoI.png", transparent=False)
