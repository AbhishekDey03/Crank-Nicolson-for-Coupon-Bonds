import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bond_Solver import BondPDESolver
from scipy.interpolate import RegularGridInterpolator, CubicSpline

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,       # Title font size
    "axes.labelsize": 16,       # Axis label font size
    "legend.fontsize": 14,      # Legend font size
    "xtick.labelsize": 14,      # X-tick font size
    "ytick.labelsize": 14,       # Y-tick font size
    "axes.linewidth": 1.2,
    
})


i_max = 100
j_max = 100
r_max_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

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

B_dirichlet_list = []
B_neumann_list = []

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axs = axs.flatten()

for idx, (r_max, ax) in enumerate(zip(r_max_values, axs)):
    bond_solver_dirichlet = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                          r_max=r_max, left_boundary_method='method4',
                                          right_boundary_method='method1')
    bond_solver_neumann = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                         r_max=r_max, left_boundary_method='method4',
                                         right_boundary_method='method2')

    bond_solver_dirichlet.solve()
    bond_solver_neumann.solve()
    B_dirichlet, r, t = bond_solver_dirichlet.get_solution_grid()
    B_neumann, _, _ = bond_solver_neumann.get_solution_grid()

    # Store boundary point values
    B_dirichlet_list.append(B_dirichlet[:, 0])
    B_neumann_list.append(B_neumann[:, 0])

    # Plot
    ax.plot(r, B_dirichlet[:, 0], label='Dirichlet',linewidth=1.5)
    ax.plot(r, B_neumann[:, 0], label='Neumann',linewidth=1.5)
    ax.set_xlabel(r'$r$')
    if idx % 3 == 0:
        # All plots will share the same yticks, so only set the ylabel for the first column
        ax.set_ylabel(r'$B(r,t=0)$')
    ax.set_title(fr'{labels[idx]} $r_\mathrm{{max}} = {r_max}$')
    ax.grid(ls=':')

# Plot formatting
fig.suptitle(rf'Bond Price at $t=0$ with $i_\mathrm{{max}}={i_max}$, $j_\mathrm{{max}}={j_max}$', fontsize=20)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.savefig('Boundary_Condition_rmax_dependency.pdf', dpi=300)
plt.show()
