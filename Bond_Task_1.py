import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bond_Solver import BondPDESolver
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline


i_max = 100
j_max = 100
r_max = 1

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

bond_solver = BondPDESolver(params, i_max=i_max, j_max=j_max, r_max=r_max,left_boundary_method='method4',right_boundary_method='method1')
bond_solver.solve()
print(f'The value of the bond at t=0, at r_0 = 0.0238, RHS B=0: {bond_solver.B_r0}')

bond_solver_actual_gridpoint = BondPDESolver(params, i_max=i_max, j_max=j_max*100, r_max=r_max,left_boundary_method='method4',right_boundary_method='method1')
bond_solver_actual_gridpoint.solve()
print(f'The value of the bond at t=0, at r_0 = 0.0238, RHS B=0,j_max=10000: {bond_solver_actual_gridpoint.B_r0}')


B,r,t = bond_solver.get_solution_grid()

interpolator = RegularGridInterpolator((r, t), B,)

r_target = params['r0']
t_target = 0
interpolated_price = interpolator((r_target, t_target))
print(f'The interpolated price of the bond at t=0, at r_0 = 0.0238, RHS B=0: {interpolated_price}')
