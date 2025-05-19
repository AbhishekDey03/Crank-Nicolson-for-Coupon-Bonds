import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_banded
from numba import njit

"""
This class handles the case of a bond following a stochastic interest rate. 
This method uses Crank-Nicolson method for the PDE solver.
In R: The LHS Boundary Condition is handled multiple different ways, and the RHS Boundary is handled with the Crank-Nicolson terms.
The Final condition is a constant, face value.
"""

@njit
def thomas_solver_nb(a, b, c, d):
    n = d.shape[0]
    beta = np.zeros(n)
    D    = np.zeros(n)
    B    = np.zeros(n)

    # Forward elimination
    beta[0] = b[0]
    D[0]    = d[0]
    for j in range(1, n):
        w       = a[j] / beta[j-1]
        beta[j] = b[j] - w * c[j-1]
        D[j]    = d[j] - w * D[j-1]

    # Back substitution
    B[n-1] = D[n-1] / beta[n-1]
    for j in range(n-2, -1, -1):
        B[j] = (D[j] - c[j] * B[j+1]) / beta[j]

    return B


class BondPDESolver:
    def __init__(self,params,i_max=100,j_max=100,r_max=1.,r_min=0,left_boundary_method=None,right_boundary_method=None):
        """
        Initialize PDE solver parameters and grids.

        params: dict containing
            kappa, theta, mu, sigma, beta, alpha, C, F, T, r0
        i_max, j_max: number of time- and r-steps
        r_max: maximum r-value
        """
        self.kappa = params['kappa']
        self.theta = params['theta']
        self.mu    = params['mu']
        self.sigma = params['sigma']
        self.beta  = params['beta']
        self.alpha = params['alpha']
        self.C     = params['C']  
        self.F     = params['F']   # face value
        self.T     = params['T']   # maturity
        self.r0    = params['r0']  # initial rate

        # Grid setup
        self.i_max = i_max   # time steps
        self.j_max = j_max   # space steps
        self.r_max = r_max
        self.r_min = r_min
        self.dt = self.T / self.i_max
        self.dr = (self.r_max-self.r_min) / self.j_max
        self.left_boundary_method = left_boundary_method if left_boundary_method is not None else 'method4' # Default to Crank Nicolson style
        self.right_boundary_method = right_boundary_method if right_boundary_method is not None else 'method1' # Default to Dirichlet BC- given in problem

        self.t_values = np.linspace(0, self.T, self.i_max + 1)
        self.r_values = np.linspace(self.r_min, self.r_max, self.j_max + 1)

        # The full solution surface
        self.B = np.zeros((self.j_max+1, self.i_max+1))

        # Initialise value of interest
        self.B_r0 = None
        self.B_r0_index = None

    def get_solution_grid(self):
        return self.B,self.r_values,self.t_values
    
    def plot_surface(self):
        """
        Plot the bond price surface as a 3D plot.
        The x-axis represents time t, the y-axis represents rate r,
        and the z-axis represents the bond price B(r,t).
        """
        T, R = np.meshgrid(self.t_values, self.r_values)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, R, self.B, cmap='viridis')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Rate r')
        ax.set_zlabel('Bond Price B(r,t)')
        plt.title('Bond Value Surface')
        plt.show()

    def plot_slice_r(self,time_index):
        plt.plot(self.r_values, self.B[:,time_index])
        plt.xlabel('Rate r')
        plt.ylabel('Bond Price B(r,t)')
        plt.title(f'Bond Value at Time t={self.t_values[time_index]:.2f}')
        plt.grid()
        plt.show()

    def plot_slice_t(self,r_index):
        plt.plot(self.t_values, self.B[r_index,:])
        plt.xlabel('Time t')
        plt.ylabel('Bond Price B(r,t)')
        plt.title(f'Bond Value at Interest Rate r={self.r_values[r_index]:.2f}')
        plt.grid()
        plt.show()

    def final_condition(self,r):
        return self.F

    def left_boundary(self, i, B_old, method='method4'):
        t_i = self.dt * (i+0.5)
        coupon = self.C * np.exp(-self.alpha * t_i)
        B0 = B_old[0]
        B1 = B_old[1]
        B2 = B_old[2]

        if method == 'method1':
            # One-sided difference at i*dt
            a0 = 0
            b0 = -1/self.dt - (self.kappa * self.theta * np.exp(self.mu*t_i)) / self.dr
            c0 =  (self.kappa * self.theta * np.exp(self.mu*t_i)) / self.dr
            d0 = - (1/self.dt) * B0 - coupon
            return a0, b0, c0, d0

        elif method == 'method2':
            # Crank-Nicolson in time, one-sided in space
            a0 = 0
            b0 = -1/self.dt - (self.kappa * self.theta * np.exp(self.mu*t_i)) / (2*self.dr)
            c0 =  (self.kappa * self.theta * np.exp(self.mu*t_i)) / (2*self.dr)
            d0 = - ((self.kappa * self.theta * np.exp(self.mu*t_i)) / (2*self.dr)) * B1 \
                - (1/self.dt - (self.kappa * self.theta * np.exp(self.mu*t_i)) / (2*self.dr)) * B0 \
                - coupon
            return a0, b0, c0, d0

        elif method == 'method3':
            """
            This method has been shelved, the reliance on t= (i+2) leads to errors
            """
            # Centered difference with ghost point using B^{i+2}_0 from self.B

            a0 = 0
            b0 = -0.5 * self.dt
            c0 = 0
            d0 = - (1/self.dt) * B0 + (1/(2*self.dt)) * B2 \
                - 0.5 * self.C * (np.exp(-self.alpha * t_i) - np.exp(-self.alpha * ((i+1)*self.dt)))
            return a0, b0, c0, d0

        elif method == 'method4':
            # Crank-Nicolson style with three-node formulation
            a_0 = -1/self.dt - (3*self.kappa*self.theta*np.exp(self.mu*t_i)) / (4*self.dr)
            b_0 = self.kappa * self.theta * np.exp(self.mu*t_i)/self.dr
            c_0 = - self.kappa * self.theta * np.exp(self.mu*t_i)/(4*self.dr)
            d_0 = - B0 * (1/self.dt - (3*self.kappa*self.theta*np.exp(self.mu*t_i)) / (4*self.dr)) \
                  - B1 * (self.kappa * self.theta * np.exp(self.mu*t_i)/self.dr) \
                  - B2 * ( - self.kappa * self.theta * np.exp(self.mu*t_i)/(4*self.dr)) \
                  - coupon
            
            a_1 = self.sigma**2 * self.dr**(2*self.beta)/(4*self.dr**2) \
                  - self.kappa * (self.theta*np.exp(self.mu*t_i)-self.dr)/(4*self.dr)
            b_1 = - 1/self.dt - 0.5*self.dr \
                    - self.sigma**2 * self.dr**(2*self.beta)/(2*self.dr**2)
            c_1 = self.sigma**2 * self.dr**(2*self.beta)/(4*self.dr**2) \
                  + self.kappa * (self.theta*np.exp(self.mu*t_i)-self.dr)/(4*self.dr)
            
            d_1 = -B0 * (self.sigma**2 * self.dr**(2*self.beta)/(4*self.dr**2) \
                  - self.kappa * (self.theta*np.exp(self.mu*t_i)-self.dr)/(4*self.dr)) \
                  -B1 * ( 1/self.dt - 0.5*self.dr \
                    - self.sigma**2 * self.dr**(2*self.beta)/(2*self.dr**2)) \
                  -B2 * ( self.sigma**2 * self.dr**(2*self.beta)/(4*self.dr**2) \
                  + self.kappa * (self.theta*np.exp(self.mu*t_i)-self.dr)/(4*self.dr)) \
                  - coupon
            
            at0 = 0
            bt0 = a_0-(c_0/c_1)*a_1
            ct0 = b_0-(c_0/c_1)*b_1
            dt0 = d_0-(c_0/c_1)*d_1
            return at0, bt0, ct0, dt0


    def right_boundary(self,i,B_old,method='method1'):
        """
        Returns the abssorbing boundary condition
        """
        if method == 'method1':
            # B=0 at RHS - Dirichlet BC
            ajmax = 0.
            bjmax = -1.
            cjmax = 0.
            djmax = 0.
        if method == 'method2':
            # dB/dr = 0 at RHS - Neumann BC to 2nd order
            ajm = 1
            bjm = -4
            cjm = 3
            B0 = B_old[self.j_max]
            B1 = B_old[self.j_max-1]
            B2 = B_old[self.j_max-2]
            djm= -B2 + 4*B1 - 3* B0
            t_i = self.dt * (i+0.5)

            ajm_1 = (self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(4*self.dr**2) \
                - self.kappa * (self.theta*np.exp(self.mu*t_i)-(self.j_max-1)*self.dr)/(4*self.dr)
            bjm_1 = -1/self.dt - 0.5*(self.j_max-1)*self.dr \
                - (self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(2*self.dr**2)
            cjm_1 = (self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(4*self.dr**2) \
                + self.kappa * (self.theta*np.exp(self.mu*t_i)-(self.j_max-1)*self.dr)/(4*self.dr)
            djm_1 = -B2*((self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(4*self.dr**2) \
                - self.kappa * (self.theta*np.exp(self.mu*t_i)-(self.j_max-1)*self.dr)/(4*self.dr)) \
                -B1*(1/self.dt - 0.5*(self.j_max-1)*self.dr \
                - (self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(2*self.dr**2)) \
                -B0*((self.sigma**2 * ((self.j_max-1)*self.dr)**(2*self.beta))/(4*self.dr**2) \
                + self.kappa * (self.theta*np.exp(self.mu*t_i)-(self.j_max-1)*self.dr)/(4*self.dr))
            
            at_jm = bjm - (ajm/ajm_1)* bjm_1
            bt_jm = cjm - (ajm/ajm_1)*cjm_1
            ct_jm = 0
            dt_jm = djm - (ajm/ajm_1)*djm_1
            
            ajmax = at_jm
            bjmax = bt_jm
            cjmax = ct_jm
            djmax = dt_jm
        return ajmax,bjmax,cjmax,djmax


    def assemble_system(self,i,B_old):
        """
        Build the tridiagonal system, A * vNew = d for the Crank Nicolson scheme.
        Parameters:
            i : the curernt time index being solved for
            B_old: contains the known solution at i+1
        Returns:
            (a,b,c,d): 1D numpy arrays of self.j_max+1 entries 
            a: sub-diagonal
            b: diagonal
            c: super-diagonal
            d: RHS
        """
        # Unpack the variables for ease of coding
        dt = self.dt
        dr = self.dr
        sigma=self.sigma
        kappa = self.kappa
        beta = self.beta
        theta = self.theta
        mu = self.mu
        t_i = (i +0.5)* dt
        # Define some useful values
        theta_exp = theta * np.exp(mu*t_i)
        sigma2 = sigma**2
        coupon = self.C * np.exp(-self.alpha*t_i)

        # Create the arrays for each of the arrays
        a = np.zeros(self.j_max + 1)
        b = np.zeros(self.j_max + 1)
        c = np.zeros(self.j_max + 1)
        d = np.zeros(self.j_max + 1)

        # Impose the LHS boundary conditions
        if self.left_boundary_method is not None:
            a[0],b[0],c[0],d[0] = self.left_boundary(i,B_old,method=self.left_boundary_method)
        else:
            a[0],b[0],c[0],d[0] = self.left_boundary(i,B_old)

        for j in range(1,self.j_max):
            r_j = j* dr
            a_j = (sigma2 * (r_j**(2*beta)))/(4*dr**2) \
                - (kappa * (theta_exp-r_j))/(4*dr)
            b_j = - 1/dt\
                - 0.5*r_j\
                - (sigma2 * (r_j**(2*beta)))/(2*dr**2)
            c_j = (sigma2 * (r_j**(2*beta)))/(4*dr**2) \
                + (kappa * (theta_exp-r_j))/(4*dr)
            
            d_j = -((sigma2 * (r_j**(2*beta)))/(4*dr**2)
                - (kappa * (theta_exp-r_j))/(4*dr) ) * B_old[j-1] \
                - (1/dt - 0.5*r_j -
                (sigma2 * (r_j**(2*beta)))/(2*dr**2)) * B_old[j] \
                - ((sigma2 * (r_j**(2*beta)))/(4*dr**2)
                    + (kappa * (theta_exp-r_j))/(4*dr)) * B_old[j+1]\
                - coupon
            a[j] = a_j
            b[j] = b_j
            c[j] = c_j
            d[j] = d_j

        # Impose the RHS boundary
        if self.right_boundary_method is not None:
            a[self.j_max],b[self.j_max],c[self.j_max],d[self.j_max] = \
                self.right_boundary(i,B_old,method=self.right_boundary_method)
        else:
            a[self.j_max],b[self.j_max],c[self.j_max],d[self.j_max] = \
            self.right_boundary(i,B_old)
        return a,b,c,d
    
    
    def thomas_solver(self,a,b,c,d):
        n = len(d)
        # Create arrays for the modified coefficients
        beta = np.zeros(n)
        D = np.zeros(n)
        B = np.zeros(n)

        # Forward elimination of the lower triangular components
        beta[0] = b[0]
        D[0] = d[0]
        for j in range(1,n):
            w = a[j]/beta[j-1]
            beta[j] = b[j] - w * c[j-1]
            D[j] = d[j] - w * D[j-1]

        # Backward substitution to get values B
        B[-1] = D[-1] / beta[-1]
        for j in reversed(range(n - 1)):
            B[j] = (D[j] - c[j] * B[j + 1]) / beta[j]

        return B

    def solve_tridiagonal_scipy(self, a, b, c, d):
        """
        Solve the tridiagonal system using SciPy's solve_banded, which solves via LU decomposition directly.
        """
        n = len(d)
        ab = np.zeros((3, n))
        ab[0,1:]   = c[:-1]
        ab[1,:]    = b
        ab[2,:-1]  = a[1:]
        return solve_banded((1,1), ab, d) # 1 sub diagonal and one sper diagonal

    def solve(self,use_scipy=False,do_return = True):
        B_old = np.zeros(self.j_max+1)
        for j in range(self.j_max+1):
            B_old[j] = self.final_condition(self.r_values[j])
        # Store the sllice in the array to plot
        self.B[:,self.i_max] = B_old.copy()
        # Step time backwards
        for i in reversed(range(self.i_max)):
            a,b,c,d = self.assemble_system(i,B_old)
            if use_scipy:
                B_new = self.solve_tridiagonal_scipy(a,b,c,d)
            else:
                B_new = thomas_solver_nb(a,b,c,d)
            
            # Overwrite for the next iteration
            B_old[:] = B_new[:].copy()
            self.B[:,i] = B_old.copy()

        # To get the value at r0, get the nearest index:
        r_index = int(self.r0/self.dr)
        self.B_r0 = self.B[r_index,0]
        self.B_r0_index = r_index
        if do_return:
            return self.B


