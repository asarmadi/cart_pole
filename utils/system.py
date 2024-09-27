import numpy as np
import scipy.optimize as opt

class CartPole:
    def __init__(self, config, xf):
        self.config = config
        self.xf     = xf

    # Dynamics of the cart-pole system
    def dynamics(self, state, action):
        x, x_dot, theta, theta_dot = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Equations of motion
        #temp = (action + m_pole * l * theta_dot**2 * sin_theta) / (m_cart + m_pole)
        #theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4/3 - m_pole * cos_theta**2 / (m_cart + m_pole)))
        #x_acc = temp - m_pole * l * theta_acc * cos_theta / (m_cart + m_pole)
    
        x_acc = (1/(self.config.m_cart+self.config.m_pole*sin_theta**2))*(action+self.config.m_pole*sin_theta*(self.config.l*theta_dot**2+self.config.g*cos_theta))
        theta_acc = (1/(self.config.l*(self.config.m_cart+self.config.m_pole*sin_theta**2)))*(-action*cos_theta-self.config.m_pole*self.config.l*theta_dot**2*cos_theta*sin_theta-(self.config.m_cart+self.config.m_pole)*self.config.g*sin_theta)
        
        return np.array([x_dot, x_acc, theta_dot, theta_acc])
    
    # Runge-Kutta 4th order method
    def step(self, state, action):
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * k1 * self.config.dt, action)
        k3 = self.dynamics(state + 0.5 * k2 * self.config.dt, action)
        k4 = self.dynamics(state + k3 * self.config.dt, action)
        return state + (k1 + 2*k2 + 2*k3 + k4) * self.config.dt / 6

    def cost(self, u, *args):
        x0 = args[0]
        cost_val = 0
        for i in range(0,self.horizon):
            
            cost_val += ( np.linalg.norm(x0-self.xf)**2 + u[i]**2 )
            x0 = self.step(x0,u[i])

        return cost_val

    def init_controller(self, xf, horizon):
        self.horizon = horizon
        self.xf  = xf
        self.umin = [-20.]
        self.umax = [20.]

    def controller(self, x0):
        
        bounds = ((self.umin[0], self.umax[0]))

        initial_guess = np.zeros(self.horizon)

        U = opt.minimize(self.cost, initial_guess, args=x0, method='SLSQP',
                         options={'maxiter': 200, 'disp': False})
        U = U.x
        
        return U[0]