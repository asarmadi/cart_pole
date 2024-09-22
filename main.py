import numpy as np
from utils.config import Config
from utils.system import CartPole
from utils.visualization import CartPoleVisualizer


# Initialize state and storage
state = np.array([0, 0, np.pi+0.1, 0])  # [x, x_dot, theta, theta_dot]
history = []
config = Config()
system = CartPole(config)

# Simulate the dynamics
for t in np.arange(0, config.T, config.dt):
    action = 0  # no control input for simplicity
    state = system.step(state, action)
    history.append(state)

history = np.array(history)

visualize_obj = CartPoleVisualizer(history, config)
visualize_obj.gen_animation()

