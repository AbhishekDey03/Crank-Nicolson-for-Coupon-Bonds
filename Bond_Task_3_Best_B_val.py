from Bond_Solver import BondPDESolver

j_max = 20000
r_max = 4.0
i_max=2200
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

bond_solver = BondPDESolver(params, i_max=i_max, j_max=j_max,
                                 r_max=r_max, left_boundary_method='method4',
                                 right_boundary_method='method2')

bond_solver.solve()
print(f'The value of the bond at t=0, at r_0 = 0.0238, RHS Neumann: {bond_solver.B_r0}')