import numpy as np
from utils.config import Config
from utils.system import CartPole
from utils.visualization import CartPoleVisualizer

from scipy.optimize import minimize


# Initialize state and storage
x0          = np.array([0, 0, np.pi+0.1, 0])  # [x, x_dot, theta, theta_dot]
xf          = np.array([0, 0, np.pi, 0]) 
mpc_horizon = 100

history = []
config = Config()
system = CartPole(config, xf)
system.init_controller(xf, mpc_horizon)

# Simulate the dynamics
for t in np.arange(0, config.T, config.dt):
    action = system.controller(x0)
    x0 = system.step(x0, action)
    history.append(x0)

history = np.array(history)

visualize_obj = CartPoleVisualizer(history, config)
visualize_obj.gen_animation()

