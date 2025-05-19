import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bond_Solver import BondPDESolver

# Booleans to activate certain things
compare_lhs_boundaries = False
compare_rhs_boundaries = True
plot_lhs_boundaries = False
plot_rhs_boundaries = True
convergence = False
i_max = 1000
j_max = 1000
r_max = 1

params = {
    'T': 3,
    'F': 240,
    'theta': 0.0289,
    'r0': 0.023,
    'kappa': 0.09389,
    'mu': 0.0141,
    'C': 10.2,
    'alpha': 0.01,
    'beta': 0.418,
    'sigma': 0.116,
}

if compare_lhs_boundaries:
    PDESolver_method1 = BondPDESolver(params) # One sided forward difference
    PDESolver_method1.solve(do_return= False)
    print(f'The value of the bond at t=0, at r_0 = 0.0238, One sided forward difference, at (i+1/2) * dt: {PDESolver_method1.B_r0:.6f}')

    PDESolver_method2 = BondPDESolver(params,left_boundary_method='method2') # Forward difference at both i and i+1 for d/dr
    PDESolver_method2.solve(do_return= False)
    print(f'The value of the bond at t=0, at r_0 = 0.0238, Forward difference at both i and i+1: {PDESolver_method2.B_r0:.6f}')

    if plot_lhs_boundaries:
        PDESolver_method1.plot_surface()
        PDESolver_method1.plot_slice_r(PDESolver_method1.B_r0_index)
        PDESolver_method1.plot_slice_t(PDESolver_method1.B_r0_index)

        PDESolver_method2.plot_surface()
        PDESolver_method2.plot_slice_r(PDESolver_method2.B_r0_index)
        PDESolver_method2.plot_slice_t(PDESolver_method2.B_r0_index)

if compare_rhs_boundaries:
    PDESolver_method4 = BondPDESolver(params,left_boundary_method='method4',r_max=r_max,j_max=j_max,i_max=i_max) # Central Differences in d/dr, also at (i+1/2) * dt, closest to crank-nicolson
    PDESolver_method4.solve(do_return= False)
    print(f'The value of the bond at t=0, at r_0 = 0.0238, Central Differences in d/dr, at (i+1/2) * dt, RHS B=0: {PDESolver_method4.B_r0}')

    PDE_Solver_alt_RHS = BondPDESolver(params, left_boundary_method='method4',right_boundary_method='method2',r_max=r_max,j_max=j_max,i_max=i_max) # Central Differences in d/dr, also at (i+1/2) * dt, RHS dB/dr=0
    PDE_Solver_alt_RHS.solve(do_return= False)
    print(f'The value of the bond at t=0, at r_0 = 0.0238, Central Differences in d/dr, at (i+1/2) * dt, RHS dB/dr=0: {PDE_Solver_alt_RHS.B_r0}')


    if plot_rhs_boundaries:
        PDESolver_method4.plot_surface()
        PDESolver_method4.plot_slice_r(0)

        PDE_Solver_alt_RHS.plot_surface()
        PDE_Solver_alt_RHS.plot_slice_r(0)

# Convergence analysis for B(r0) with respect to j_max and i_max
if convergence:
    i_max_vals = [100, 200, 500]
    j_max_vals = [100, 200, 500, 1000, 2000, 3000]

    B_r0_matrix = np.zeros((len(i_max_vals), len(j_max_vals)))

    # Fill matrix with B(r0) values
    for i_idx, i_max in enumerate(i_max_vals):
        for j_idx, j_max in enumerate(j_max_vals):
            PDESolver = BondPDESolver(params, left_boundary_method='method4',
                                    right_boundary_method='method1',
                                    r_max=r_max, j_max=j_max, i_max=i_max)
            PDESolver.solve(do_return=False)
            B_r0_matrix[i_idx, j_idx] = PDESolver.B_r0

    # Plot convergence for each i_max (varying j_max)
    for i_idx, i_max in enumerate(i_max_vals):
        plt.figure()
        plt.plot(j_max_vals, B_r0_matrix[i_idx, :], marker='o')
        plt.xlabel('j_max')
        plt.ylabel('B(r0)')
        plt.title(f'Convergence of B(r0) vs j_max (i_max={i_max})')
        plt.grid(True)
        plt.show()
