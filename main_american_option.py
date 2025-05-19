import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bond_Solver import BondPDESolver
from American_Option_Solver import AmericanOptionSolver

i_max = 100
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


bond_solver = BondPDESolver(bond_params,i_max=i_max,j_max=j_max,r_max=r_max)

bond_surface = bond_solver.solve(do_return=True)

option_params['B'] = bond_surface

option_solver = AmericanOptionSolver(option_params,i_max=i_max,j_max=j_max,r_max=r_max)

option_surface = option_solver.solve()

bond_solver.plot_surface()
option_solver.plot_surface()

option_solver.plot_slice_r(0)
option_solver.plot_slice_r(-1)