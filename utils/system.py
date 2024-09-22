import numpy as np

class CartPole:
    def __init__(self, config):
        self.config = config

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