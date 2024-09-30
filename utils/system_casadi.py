import casadi
from casadi import SX, DM
from utils.system import BaseSystem

class CartPoleCasadi(BaseSystem):
    def __init__(self, config, xf):
        super().__init__(config)
        self.config          = config
        self.index_x         = 0
        self.index_x_dot     = 1
        self.index_theta     = 2
        self.index_theta_dot = 3
        self.xf              = xf

        ## Problem size
        self.n_states = 4
        self.n_inputs = 1
        ## Create Optimization Variables
        self.delta_t = SX.sym('delta_t')
        self.states  = SX.sym('state', self.config.mpc_horizon, self.n_states)
        self.inputs  = SX.sym('input', self.config.mpc_horizon-1, self.n_inputs)
        ## Create mapping between structured and flattened variables
        variables_list = [self.states, self.inputs]
        variables_name = ['states', 'inputs']
        self.variables_flat = casadi.vertcat(*[casadi.reshape(e,-1,1) for e in variables_list])
        self.pack_variables_fn = casadi.Function('pack_variables_fn', variables_list, [self.variables_flat], variables_name, ['flat'])
        self.unpack_variables_fn = casadi.Function('unpack_variables_fn', [self.variables_flat], variables_list, ['flat'], variables_name)


    def dynamics_diff(self, x_, u):
        x           = x_[:,self.index_x]
        x_dot       = x_[:,self.index_x_dot]
        theta       = x_[:,self.index_theta]
        theta_dot   = x_[:,self.index_theta_dot]

        sin_theta = casadi.sin(theta)
        cos_theta = casadi.cos(theta)

        action = u[:, 0]

        x_acc = (1/(self.config.m_cart+self.config.m_pole*sin_theta**2))*(action+self.config.m_pole*sin_theta*(self.config.l*theta_dot**2+self.config.g*cos_theta))
        theta_acc = (1/(self.config.l*(self.config.m_cart+self.config.m_pole*sin_theta**2)))*(-action*cos_theta-self.config.m_pole*self.config.l*theta_dot**2*cos_theta*sin_theta-(self.config.m_cart+self.config.m_pole)*self.config.g*sin_theta)

        dxdt = 0*x_
        dxdt[:,self.index_x]         = x_dot
        dxdt[:,self.index_x_dot]     = x_acc
        dxdt[:,self.index_theta]     = theta_dot
        dxdt[:,self.index_theta_dot] = theta_acc
        return dxdt

    def controller(self, x0):
        ## Box constraints
        lower_bounds = self.unpack_variables_fn(flat=-float('inf'))
        upper_bounds = self.unpack_variables_fn(flat=float('inf'))
        lower_bounds['inputs'][:,0] = -20.0 # Minimum force
        upper_bounds['inputs'][:,0] = 20.0 # Maximum force
        ## Initial state
        lower_bounds['states'][0, self.index_x]           = x0[0]
        upper_bounds['states'][0, self.index_x]           = x0[0]
        lower_bounds['states'][0, self.index_x_dot]       = x0[1]
        upper_bounds['states'][0, self.index_x_dot]       = x0[1]
        lower_bounds['states'][0, self.index_theta]       = x0[2]
        upper_bounds['states'][0, self.index_theta]       = x0[2]
        lower_bounds['states'][0, self.index_theta_dot]   = x0[3]
        upper_bounds['states'][0, self.index_theta_dot]   = x0[3]
        ## Final state
        lower_bounds['states'][-1,self.index_x]           = self.xf[0]
        upper_bounds['states'][-1,self.index_x]           = self.xf[0]
        lower_bounds['states'][-1,self.index_x_dot]       = self.xf[1]
        upper_bounds['states'][-1,self.index_x_dot]       = self.xf[1]
        lower_bounds['states'][-1,self.index_theta]       = self.xf[2]
        upper_bounds['states'][-1,self.index_theta]       = self.xf[2]
        lower_bounds['states'][-1,self.index_theta_dot]   = self.xf[3]
        upper_bounds['states'][-1,self.index_theta_dot]   = self.xf[3]

        X0 = self.states[0:self.config.mpc_horizon-1,:]
        X1 = self.states[1:self.config.mpc_horizon,:]
        # Heun's method (some other method, like RK4 could also be used here)
        K1 = self.dynamics_diff(X0, self.inputs)
        K2 = self.dynamics_diff(X0 + self.config.dt * K1, self.inputs)
        defect = X0 + self.config.dt*(K1+K2)/2.0 - X1
        defect = casadi.reshape(defect, -1, 1)
        ## Optimization objective
        # Maximize final mass
        objective = casadi.sum1(casadi.sum2(self.inputs**2))

        solver = casadi.nlpsol('solver', 'ipopt', {'x':self.variables_flat, 'f':objective, 'g':defect,'verbose':False})
        result = solver(x0=1.0, lbg=0.0, ubg=0.0,
                lbx=self.pack_variables_fn(**lower_bounds)['flat'],
                ubx=self.pack_variables_fn(**upper_bounds)['flat'])
        results = self.unpack_variables_fn(flat=result['x'])
        print(results['inputs'][0])
        print(type(results['inputs'][0]))

        return results['inputs'][0].__float__()

