import numpy as np
import matplotlib.pyplot as plt
from Bond_Solver import BondPDESolver
from American_Option_Solver import AmericanOptionSolver
r_max_run = False
i_max_run=False
j_max_run=False
best_estimate = True

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

base_bond_params = {
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

base_option_params = {
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

if i_max_run:
    i_max_vals = [100, 200, 400, 800, 1600, 2200, 3200]
    V_r0_list = []

    for i_max in i_max_vals:
        print(f"Running i_max = {i_max}")
        
        # Recalculate Bond Surface
        bond_solver = BondPDESolver(base_bond_params, i_max=i_max, j_max=20000, r_max=4.0)
        B_surface = bond_solver.solve(do_return=True)

        # Setup option parameters
        option_params = base_option_params.copy()
        option_params['B'] = B_surface

        # Solve American Option
        option_solver = AmericanOptionSolver(option_params, i_max=i_max, j_max=20000, r_max=4.0)
        option_solver.solve(do_return=False)
        V_r0_list.append(option_solver.V_r0)

    plt.plot(i_max_vals, V_r0_list, marker='o')
    plt.xlabel(r'$i_{\mathrm{max}}$')
    plt.ylabel(r'$V(r_0, 0)$')
    plt.title('Convergence with Time Steps')
    plt.grid(True)
    plt.savefig('convergence_time_steps.pdf', dpi=300)
    plt.close()
    print("Convergence with Time Steps finished")

if r_max_run:
    r_max_vals = np.arange(1.,4.,0.1)
    V_r0_list = []

    for r_max in r_max_vals:
        print(f"Running r_max = {r_max}")

        # Recalculate Bond Surface
        bond_solver = BondPDESolver(base_bond_params, i_max=1000, j_max=1000, r_max=r_max)
        B_surface = bond_solver.solve(do_return=True)

        # Setup option parameters
        option_params = base_option_params.copy()
        option_params['B'] = B_surface

        # Solve American Option
        option_solver = AmericanOptionSolver(option_params, i_max=1000, j_max=1000, r_max=r_max)
        option_solver.solve(do_return=False)
        V_r0_list.append(option_solver.V_r0)

    plt.plot(r_max_vals, V_r0_list, marker='o')
    plt.xlabel(r'$r_{\mathrm{max}}$')
    plt.ylabel(r'$V(r_0, 0)$')
    plt.title('Effect of Domain Size')
    plt.grid(True)
    plt.savefig('effect_domain_size.pdf', dpi=300)
    print("Effect of Domain Size finished")

if j_max_run:
    j_max_vals = [100, 200, 400, 800, 1600, 3200,6400,12800,20000]
    V_r0_list = []

    for j_max in j_max_vals:
        print(f"Running j_max = {j_max}")

        # Recalculate Bond Surface
        bond_solver = BondPDESolver(base_bond_params, i_max=2000, j_max=j_max, r_max=4.0)
        B_surface = bond_solver.solve(do_return=True)

        # Setup option parameters
        option_params = base_option_params.copy()
        option_params['B'] = B_surface

        # Solve American Option
        option_solver = AmericanOptionSolver(option_params, i_max=2000, j_max=j_max, r_max=4.0)
        option_solver.solve(do_return=False)
        V_r0_list.append(option_solver.V_r0)

    plt.plot(j_max_vals, V_r0_list, marker='o')
    plt.xlabel(r'$j_{\mathrm{max}}$')
    plt.ylabel(r'$V(r_0, 0)$')
    plt.title('Convergence with Space Steps')
    plt.grid(True)
    plt.savefig('convergence_space_steps.pdf', dpi=300)
    plt.close()
    print("Convergence with Space Steps finished")

if best_estimate:
    bond_solver = BondPDESolver(base_bond_params, i_max=2000, j_max=20000, r_max=4.0)
    B_surface = bond_solver.solve(do_return=True)
    option_params = base_option_params.copy()
    option_params['B'] = B_surface
    option_solver = AmericanOptionSolver(option_params, i_max=2000, j_max=20000, r_max=4.0)
    option_solver.solve(do_return=False)
    V, r, t = option_solver.get_solution_grid()
    V_r0= option_solver.V_r0
    print('V(r0,0) =', V_r0)
