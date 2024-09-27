
import casadi
import numpy as np
from casadi import SX, DM
from utils.config import Config
from utils.visualization import CartPoleVisualizer

config = Config()

index_x         = 0
index_x_dot     = 1
index_theta     = 2
index_theta_dot = 3


def differential_equation(x_, u):
    x           = x_[:,index_x]
    x_dot       = x_[:,index_x_dot]
    theta       = x_[:,index_theta]
    theta_dot   = x_[:,index_theta_dot]

    sin_theta = casadi.sin(theta)
    cos_theta = casadi.cos(theta)

    action = u[:, 0]

    x_acc = (1/(config.m_cart+config.m_pole*sin_theta**2))*(action+config.m_pole*sin_theta*(config.l*theta_dot**2+config.g*cos_theta))
    theta_acc = (1/(config.l*(config.m_cart+config.m_pole*sin_theta**2)))*(-action*cos_theta-config.m_pole*config.l*theta_dot**2*cos_theta*sin_theta-(config.m_cart+config.m_pole)*config.g*sin_theta)
        
    dxdt = 0*x_
    dxdt[:,index_x]         = x_dot
    dxdt[:,index_x_dot]     = x_acc
    dxdt[:,index_theta]     = theta_dot
    dxdt[:,index_theta_dot] = theta_acc
    return dxdt

## Problem size
n_states = 4
n_inputs = 1
## Create Optimization Variables
delta_t = SX.sym('delta_t')
states = SX.sym('state', config.mpc_horizon, n_states)
inputs = SX.sym('input', config.mpc_horizon-1, n_inputs)
## Create mapping between structured and flattened variables
variables_list = [states, inputs]
variables_name = ['states', 'inputs']
variables_flat = casadi.vertcat(*[casadi.reshape(e,-1,1) for e in variables_list])
pack_variables_fn = casadi.Function('pack_variables_fn', variables_list, [variables_flat], variables_name, ['flat'])
unpack_variables_fn = casadi.Function('unpack_variables_fn', [variables_flat], variables_list, ['flat'], variables_name)
## Box constraints
lower_bounds = unpack_variables_fn(flat=-float('inf'))
upper_bounds = unpack_variables_fn(flat=float('inf'))
lower_bounds['inputs'][:,0] = -20.0 # Minimum force
upper_bounds['inputs'][:,0] = 20.0 # Maximum force
## Initial state
lower_bounds['states'][0,index_x]           = 0.0
upper_bounds['states'][0,index_x]           = 0.0
lower_bounds['states'][0,index_x_dot]       = 0.0
upper_bounds['states'][0,index_x_dot]       = 0.0
lower_bounds['states'][0,index_theta]       = 0.0
upper_bounds['states'][0,index_theta]       = 0.0
lower_bounds['states'][0,index_theta_dot]   = 0.0
upper_bounds['states'][0,index_theta_dot]   = 0.0
## Final state
lower_bounds['states'][-1,index_x]           = 0.0
upper_bounds['states'][-1,index_x]           = 0.0
lower_bounds['states'][-1,index_x_dot]       = 0.0
upper_bounds['states'][-1,index_x_dot]       = 0.0
lower_bounds['states'][-1,index_theta]       = np.pi
upper_bounds['states'][-1,index_theta]       = np.pi
lower_bounds['states'][-1,index_theta_dot]   = 0.0
upper_bounds['states'][-1,index_theta_dot]   = 0.0
## Differential equation constraints
# There is no loop here, because it is vectorized.
X0 = states[0:config.mpc_horizon-1,:]
X1 = states[1:config.mpc_horizon,:]
# Heun's method (some other method, like RK4 could also be used here)
K1 = differential_equation(X0, inputs)
K2 = differential_equation(X0 + config.dt * K1, inputs)
defect = X0 + config.dt*(K1+K2)/2.0 - X1
defect = casadi.reshape(defect, -1, 1)
## Optimization objective
# Maximize final mass
objective = casadi.sum1(casadi.sum2(inputs**2))
## Run optimization
# This uses a naive initialization, it starts every variable at 1.0.
# Some problems require a cleverer initialization, but that is a story for another time.
solver = casadi.nlpsol('solver', 'ipopt', {'x':variables_flat, 'f':objective, 'g':defect})
result = solver(x0=1.0, lbg=0.0, ubg=0.0,
                lbx=pack_variables_fn(**lower_bounds)['flat'],
                ubx=pack_variables_fn(**upper_bounds)['flat'])
results = unpack_variables_fn(flat=result['x'])
history = np.array(results['states'])

visualize_obj = CartPoleVisualizer(history, config)
visualize_obj.gen_animation()
  