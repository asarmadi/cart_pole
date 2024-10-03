class Config:
    def __init__(self):
        # Define constants
        self.g = 9.8  # acceleration due to gravity, in m/s^2
        self.m_cart = 1.0  # mass of the cart, in kg
        self.m_pole = 0.1  # mass of the pole, in kg
        self.l = 1.0  # length of the pole, in meters
        self.J = (self.m_pole * self.l**2) # Moment of inertia
        self.dt = 0.04  # time step, in seconds
        self.T = 5  # total time, in seconds
        self.mpc_horizon = 150 # Horizon lenght in time step