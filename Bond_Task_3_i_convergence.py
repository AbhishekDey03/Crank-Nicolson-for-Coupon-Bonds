import numpy as np
import matplotlib.pyplot as plt
from Bond_Solver import BondPDESolver

# Plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
})

j_max = 20000
r_max = 4.0
params = {
    'T': 3,
    'F': 240,
    'theta': 0.0289,
    'r0': 0.0238,
    'kappa': 0.09389,
    'mu': 0.0141,
    'C': 10.2,
    'alpha': 0.01,
    'beta': 0.418,
    'sigma': 0.116,
}

i_max_values = [100,200,300,500,800,1200,2200,3600,6000,10000]
r_0_vals = []

for i_max in i_max_values:
    bond_solver = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                 r_max=r_max, left_boundary_method='method4',
                                 right_boundary_method='method2')
    bond_solver.solve()
    r_0_vals.append(bond_solver.B_r0)

baseline = r_0_vals[0]
normalized_vals = [val - baseline for val in r_0_vals]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(i_max_values, normalized_vals, marker='o', color='blue',
        label=r'$B(r_0,t=0)_{i_{\mathrm{max}}} - B(r_0,t=0)_{i_{\mathrm{max}}=100}$')
ax.set_xlabel(r'$i_{\mathrm{max}}$')
ax.set_ylabel(r'$B(r_0,t=0)_{i_{\mathrm{max}}} - B(r_0,t=0)_{i_{\mathrm{max}}=100}$')
ax.set_title(r'Convergence of $B(r_0,t=0)$ with varying $i_\mathrm{max}$', fontsize=20)
ax.grid(ls=':')
ax.legend()
plt.tight_layout()
plt.savefig('imax_dep_normalised.pdf', dpi=300)
plt.show()
