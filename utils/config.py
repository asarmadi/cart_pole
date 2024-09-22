class Config:
    def __init__(self):
        # Define constants
        self.g = 9.8  # acceleration due to gravity, in m/s^2
        self.m_cart = 1.0  # mass of the cart, in kg
        self.m_pole = 0.1  # mass of the pole, in kg
        self.l = 1.0  # length of the pole, in meters
        self.dt = 0.02  # time step, in seconds
        self.T = 20  # total time, in seconds