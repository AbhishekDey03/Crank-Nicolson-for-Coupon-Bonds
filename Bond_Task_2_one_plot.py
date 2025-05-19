import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bond_Solver import BondPDESolver
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


colors = sns.color_palette("colorblind", 2)

i_max, j_max = 100, 100
r_max_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0,5.0]

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

fig, ax = plt.subplots(figsize=(12, 8))

label_dirichlet = True
label_neumann = True

for r_max in r_max_values:
    bond_solver_dirichlet = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                          r_max=r_max, left_boundary_method='method4',
                                          right_boundary_method='method1')
    bond_solver_neumann = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                         r_max=r_max, left_boundary_method='method4',
                                         right_boundary_method='method2')

    bond_solver_dirichlet.solve()
    bond_solver_neumann.solve()

    B_dirichlet, r, _ = bond_solver_dirichlet.get_solution_grid()
    B_neumann, _, _ = bond_solver_neumann.get_solution_grid()

    B_dirichlet_list.append(B_dirichlet[:, 0])
    B_neumann_list.append(B_neumann[:, 0])

    ax.plot(r, B_dirichlet[:, 0], color=colors[0], alpha=0.5, label='Dirichlet' if label_dirichlet else "",linewidth=1.5)
    ax.plot(r, B_neumann[:, 0], color=colors[1], alpha=0.5, label='Neumann' if label_neumann else "",linewidth=1.5)

    label_dirichlet = False
    label_neumann = False

ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$B(r,t=0)$')
ax.set_title(r'Bond Price at $t=0$ for Varying $r_\mathrm{max}$', fontsize=20)
ax.grid(ls=':')
ax.legend()
plt.tight_layout()
plt.savefig('Boundary_Comparison_TwoColors_TwoLabels.pdf', dpi=300)
plt.show()
