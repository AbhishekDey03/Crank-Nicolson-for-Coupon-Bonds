import numpy as np
import matplotlib.pyplot as plt
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

# Fixed resolution
i_max = 1000
j_max = 1000

# Sweep over different r_max values
rmax_values = [1.0,1.5, 2.0,2.5, 3.0,3.5, 4.0]
exercise_thresholds = []

# Parameters
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

for r_max in rmax_values:
    print(f"Solving for r_max = {r_max}")
    
    bond_solver = BondPDESolver(
        bond_params,
        i_max=i_max,
        j_max=j_max,
        r_max=r_max,
        right_boundary_method='method2',
        left_boundary_method='method4'
    )
    bond_surface = bond_solver.solve(do_return=True)
    option_params['B'] = bond_surface

    option_solver = AmericanOptionSolver(option_params, i_max=i_max, j_max=j_max, r_max=r_max)
    option_surface = option_solver.solve()

    V, r_grid, t_grid = option_solver.get_solution_grid()
    V_T1 = V[:, -1]  # Value at t = T1

    exercise_indices = np.where(V_T1 > 0)[0]
    r_exercise_threshold = r_grid[exercise_indices[0]] if len(exercise_indices) > 0 else np.nan
    exercise_thresholds.append(r_exercise_threshold)


# Print LaTeX table
print("\\begin{table}[H]")
print("\\centering")
print("\\begin{tabular}{cc}")
print("\\toprule")
print("$r_\\mathrm{max}$ & Minimum $r$ where $V(r, T_1) > 0$ \\\\")
print("\\midrule")

for r_max, threshold in zip(rmax_values, exercise_thresholds):
    print(f"{r_max:.2f} & {threshold:.6f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Exercise threshold at $t = T_1$ vs domain boundary $r_\\mathrm{max}$}")
print("\\label{tab:exercise-threshold}")
print("\\end{table}")
