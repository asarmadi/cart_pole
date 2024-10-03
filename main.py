import numpy as np
from utils.config import Config
from utils.system import CartPole
from utils.system_casadi import CartPoleCasadi
from utils.visualization import CartPoleVisualizer

from scipy.optimize import minimize
from tqdm import tqdm


# Initialize state and storage
x0          = np.array([0, 0, 0, 0])  # [x, x_dot, theta, theta_dot]
xf          = np.array([0, 0, np.pi, 0]) 

states  = [x0]
actions = []
config = Config()
system = CartPoleCasadi(config, xf)
#system = CartPole(config, xf)
#system.init_controller(xf, config.mpc_horizon)

# Simulate the dynamics
for t in tqdm(np.arange(0, config.T, config.dt)):
    action = system.controller(x0)
    x0 = system.step(x0, action)
    states.append(x0)
    actions.append(action)

states = np.array(states)
print(f"States: {states.shape}")
actions = np.array(actions).reshape(-1,1)
a = np.hstack((states[:-1,:],actions))
print(a.shape)
np.savetxt("./out/out_mpc.csv", a, delimiter=",", header='x,x_dot,theta,theta_dot,u')

visualize_obj = CartPoleVisualizer(states, config)
visualize_obj.gen_animation()
visualize_obj.plot(states, actions)

