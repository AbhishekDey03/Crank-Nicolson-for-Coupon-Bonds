import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_banded
from numba import njit

"""
This class handles the case of an American option on a bond, as defined in Bond_Solver.py
This method uses Crank-Nicolson method for the PDE solver.
In R: The LHS Boundary Condition is handled with a three-point forward difference scheme, and the RHS Boundary is handled with the Crank-Nicolson terms.
The Final condition is imposed by taking in the calculated full bond surface.
"""


class AmericanOptionSolver:
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
        self.C     = params['C']  
        self.X     = params['X']   # exercise price
        self.T     = params['T']   # Time domain
        self.r0    = params['r0']  # initial rate
        self.B     = params['B']   # Option Surface
        self.T1    = params['T1']  # maturity for option

        # Grid setup
        self.i_max = i_max   # time steps
        self.j_max = j_max   # space steps
        self.r_max = r_max
        self.r_min = r_min
        self.dt = self.T / self.i_max
        self.dr = (self.r_max-self.r_min) / self.j_max
        self.left_boundary_method = left_boundary_method
        self.right_boundary_method = right_boundary_method

        self.i_T1 = int(self.T1 / self.dt) # time index for T1 for early exercise

        self.t_values = np.linspace(0, self.T1, self.i_T1 + 1)
        self.r_values = np.linspace(self.r_min, self.r_max, self.j_max + 1)

        # Initialise value of interest
        self.V = np.zeros((self.j_max+1, self.i_T1+1))
        self.V_r0 = None
        self.V_r0_index = None


    def get_solution_grid(self):
        return self.V,self.r_values,self.t_values
    
    def plot_surface(self):
        """
        Plot the bond price surface as a 3D plot.
        The x-axis represents time t, the y-axis represents rate r,
        and the z-axis represents the Option price V(r,t).
        """
        T, R = np.meshgrid(self.t_values, self.r_values)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, R, self.V, cmap='viridis')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Rate r')
        ax.set_zlabel('Option Price V(r,t)')
        plt.title('Option Value Surface')
        plt.show()

    def plot_slice_r(self,time_index):
        plt.plot(self.r_values, self.V[:,time_index])
        plt.xlabel('Rate r')
        plt.ylabel(f'Option Price V(r,t)')
        plt.title(f'Option Value at Time t={self.t_values[time_index]:.2f}')
        plt.grid()
        plt.show()

    def plot_slice_t(self,r_index):
        plt.plot(self.t_values, self.V[r_index,:])
        plt.xlabel('Time t')
        plt.ylabel('Option Price V(r,t)')
        plt.title(f'Option Value at Interest Rate r={self.r_values[r_index]:.2f}')
        plt.grid()
        plt.show()

    def final_condition(self, r_index):
        """
        Final condition for American option at each r_index.
        """
        B_T1 = self.B[r_index, self.i_T1]
        return max(self.X - B_T1, 0)

    
    def left_boundary(self,i,V_old):
        t_i = self.dt * i
        V0 = V_old[0]
        V1 = V_old[1]
        V2 = V_old[2]

        a_0 = -1/self.dt - (3*self.kappa*self.theta*np.exp(self.mu *t_i))/(4*self.dr)
        b_0 = self.kappa*self.theta*np.exp(self.mu *t_i)/(self.dr)
        c_0 = - self.kappa*self.theta*np.exp(self.mu *t_i)/(4*self.dr)
        a_1 = (self.sigma**2 * (self.dr)**(2*self.beta))/(4*self.dr**2) - self.kappa* (self.theta*np.exp(self.mu *t_i) - self.dr)/(4*self.dr)
        b_1 = -1/self.dt - (self.sigma**2 * (self.dr)**(2*self.beta))/(2*self.dr**2) - 0.5*self.dr
        c_1 = (self.sigma**2 * (self.dr)**(2*self.beta))/(4*self.dr**2) + self.kappa* (self.theta*np.exp(self.mu *t_i) - self.dr)/(4*self.dr)
        d_0 = -V0 * (1/self.dt - (3*self.kappa*self.theta*np.exp(self.mu *t_i))/(4*self.dr)) \
              -V1 * (self.kappa*self.theta*np.exp(self.mu *t_i)/(self.dr)) \
              -V2 * (- self.kappa*self.theta*np.exp(self.mu *t_i)/(4*self.dr))
        d_1 = -V0 * ((self.sigma**2 * (self.dr)**(2*self.beta))/(4*self.dr**2) - self.kappa* (self.theta*np.exp(self.mu *t_i) - self.dr)/(4*self.dr)) \
              -V1 * (1/self.dt - (self.sigma**2 * (self.dr)**(2*self.beta))/(4*self.dr**2) - 0.5*self.dr) \
              -V2 * ((self.sigma**2 * (self.dr)**(2*self.beta))/(4*self.dr**2) + self.kappa* (self.theta*np.exp(self.mu *t_i) - self.dr)/(4*self.dr))
        
        # Define the eliminated tridiagonal components
        at_0 = 0
        bt_0 = a_0 - c_0/c_1 * a_1
        ct_0 = b_0 - c_0/c_1 * b_1
        dt_0 = d_0 - c_0/c_1 * d_1




        return at_0, bt_0, ct_0, dt_0
    

    def right_boundary(self, i, V_old):
        V_jmax = V_old[self.j_max]
        a_jmax = 0.0
        b_jmax = 0.5
        c_jmax = 0.0
        d_jmax = self.X - self.B[self.j_max, i] - 0.5 * V_jmax
        return a_jmax, b_jmax, c_jmax, d_jmax

    def assemble_system(self,i,V_old):
        """
        Build the tridiagonal system, A * vNew = d for the Crank Nicolson scheme.
        Parameters:
            i : the curernt time index being solved for
            V_old: contains the known solution at i+1
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
        t_i = i * dt
        # Useful Values
        theta_exp = theta * np.exp(mu*t_i)
        sigma2 = sigma**2
        twobeta = 2 * beta

        # Create the arrays for each of the arrays
        a = np.zeros(self.j_max + 1)
        b = np.zeros(self.j_max + 1)
        c = np.zeros(self.j_max + 1)
        d = np.zeros(self.j_max + 1)
       
        a[0], b[0], c[0], d[0] = self.left_boundary(i,V_old)

        for j in range(1,self.j_max):
            r_j = j*dr

            a[j] = (sigma2* r_j**twobeta)/(4*dr**2) - kappa * (theta_exp - r_j)/(4*dr)
            b[j] = -1/dt - (sigma2 * r_j**twobeta)/(2*dr**2) - 0.5*r_j
            c[j] = (sigma2* r_j**twobeta)/(4*dr**2) + kappa * (theta_exp - r_j)/(4*dr)
            d[j] = -V_old[j-1] * ( (sigma2* r_j**twobeta)/(4*dr**2) -  kappa * (theta_exp - r_j)/(4*dr)) \
                   -V_old[j] * (1/dt - (sigma2 * r_j**twobeta)/(2*dr**2) - 0.5*r_j) \
                   -V_old[j+1] * ( (sigma2* r_j**twobeta)/(4*dr**2) + kappa * (theta_exp - r_j)/(4*dr))
        
        a[self.j_max],b[self.j_max],c[self.j_max],d[self.j_max] = \
            self.right_boundary(i,V_old)
        
        return a,b,c,d
    
    def solve(self, do_return=True):
        V_old = np.zeros(self.j_max+1)
        for j in range(self.j_max+1):
            V_old[j] = self.final_condition(j)

        self.V[:, self.i_T1] = V_old.copy()  # Set final time condition

        # PSOR parameters
        omega = 1.2  # Relaxation parameter
        tol = 1e-8   # Tolerance for convergence
        max_iter = 10000

        for i in reversed(range(self.i_T1)):
            a, b, c, d = self.assemble_system(i, V_old)
            
            # Initial guess
            V_new = V_old.copy()

            # PSOR iteration
            for iteration in range(max_iter):
                error = 0.0
                for j in range(self.j_max+1):
                    if j == 0:
                        # Left boundary
                        y = (d[j] - c[j]*V_new[j+1]) / b[j]
                    elif j == self.j_max:
                        # Right boundary
                        y = (d[j] - a[j]*V_new[j-1]) / b[j]
                    else:
                        # Interior points
                        y = (d[j] - a[j]*V_new[j-1] - c[j]*V_new[j+1]) / b[j]
                    
                    # Relaxation
                    y = V_new[j] + omega * (y - V_new[j])

                    exercise_value = self.X - self.B[j, i] # Exercise value
                    y = max(y, exercise_value)

                    # Calculate error (residual)
                    error += (y - V_new[j])**2

                    # Update
                    V_new[j] = y

                # Convergence check
                if error < tol**2:
                    break
            else:
                print(f"Warning: PSOR not converged at time step {i}, residual error = {error:.2e}")

            V_old = V_new.copy()
            self.V[:, i] = V_old.copy()
            self.V_r0_index = np.argmin(np.abs(self.r_values - self.r0))
            self.V_r0 = self.V[self.V_r0_index, 0]

        if do_return:
            return self.V