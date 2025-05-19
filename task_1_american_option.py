import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bond_Solver import BondPDESolver
from American_Option_Solver import AmericanOptionSolver

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

i_max = 1000
j_max = 1000
r_max = 1

bond_params = {
    'kappa': 0.09389,
    'theta': 0.0289,
    'mu': 0.0141,
    'sigma': 0.116,
    'beta': 0.418,
    'alpha': 0.01,
    'C': 10.2,
    'F': 240,
    'T': 3,
    'r0': 0.0238
}

option_params = {
    'kappa': 0.09389,
    'theta': 0.0289,
    'mu': 0.0141,
    'sigma': 0.116,
    'beta': 0.418,
    'C': 10.2,
    'X': 245,
    'T': 3,
    'T1': 1.02,
    'r0': 0.0238,
    'B': None  
}

#Neumann bond solver
bond_solver = BondPDESolver(bond_params,i_max=i_max,j_max=j_max,r_max=r_max,right_boundary_method='method2',left_boundary_method='method4')

bond_surface = bond_solver.solve(do_return=True)

option_params['B'] = bond_surface

option_solver = AmericanOptionSolver(option_params,i_max=i_max,j_max=j_max,r_max=r_max)

option_surface = option_solver.solve()

V,r,t = option_solver.get_solution_grid()

V_t0 = V[:,0]
V_T1 = V[:,-1]

exercise_indices = np.where(V_T1 > 0)[0]
r_exercise_threshold = r[exercise_indices[0]] if len(exercise_indices) > 0 else None

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot V(r, t=0)
axs[0].plot(r, V_t0, label=r'$V(r, t=0)$')
axs[0].set_xlabel(r'$r$')
axs[0].set_ylabel(r'$V(r,t)$')
axs[0].set_title(r'American Option Value at $t = 0$')
axs[0].grid(True)
axs[0].legend()

# Plot V(r, t=T1)
axs[1].plot(r, V_T1, label=r'$V(r, t=T_1)$')
axs[1].set_xlabel(r'$r$')
axs[1].set_title(r'American Option Value at $t = T_1$')
axs[1].grid(True)
axs[1].legend()

fig.suptitle(r'American option $V(r,t=0,T_1)$ $i_\mathrm{max}=j_\mathrm{max}=1000$', fontsize=20)
# Add horizontal line at exercise boundary
if r_exercise_threshold is not None:
    axs[1].axvline(x=r_exercise_threshold, color='red', linestyle='--', label=f'Exercise threshold: r = {r_exercise_threshold:.3f}')
    axs[1].legend()

plt.tight_layout()
plt.savefig('american_option_exercise_boundary.pdf', dpi=300)
print(r_exercise_threshold)
plt.show()