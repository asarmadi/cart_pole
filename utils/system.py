import casadi
from casadi import SX, DM
from utils.visualization import CartPoleVisualizer
import numpy as np

class CartPole:
    def __init__(self, config, xf):
        self.config          = config
        self.index_x         = 0
        self.index_x_dot     = 1
        self.index_theta     = 2
        self.index_theta_dot = 3
        self.xf              = xf
        self.init_guess      = 1

        ## Problem size
        self.n_states = 4
        self.n_inputs = 1
        
    def cal_energy(self):
        x           = self.states[:,self.index_x]
        x_dot       = self.states[:,self.index_x_dot]
        theta       = self.states[:,self.index_theta]
        theta_dot   = self.states[:,self.index_theta_dot]

        pot = -self.config.m_pole*self.config.g*self.config.l*casadi.cos(theta)
        kin =  0.5*(self.config.m_cart+self.config.m_pole)*x_dot**2
        kin += self.config.m_pole*x_dot*theta_dot*self.config.l*casadi.cos(theta)
        kin += 0.5*self.config.J*theta_dot**2
        return pot, kin

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

    def step(self, x_, u_):
        K1 = self.dynamics_diff(x_, u_)
        K2 = self.dynamics_diff(x_ + 0.5 * self.config.dt  * K1, u_)
        K3 = self.dynamics_diff(x_ + 0.5 * self.config.dt  * K2, u_)
        K4 = self.dynamics_diff(x_ + self.config.dt  * K3, u_)
        defect = x_ + self.config.dt *(K1+2*K2+2*K3+K4)/6.0
        return defect


    def controller(self, x0):
        ## Create Optimization Variables
        self.states  = SX.sym('state', self.config.mpc_horizon, self.n_states)
        self.inputs  = SX.sym('input', self.config.mpc_horizon-1, self.n_inputs)
        ## Create mapping between structured and flattened variables
        variables_list           = [self.states, self.inputs]
        variables_name           = ['states', 'inputs']
        self.variables_flat      = casadi.vertcat(*[casadi.reshape(e,-1,1) for e in variables_list])
        self.pack_variables_fn   = casadi.Function('pack_variables_fn', variables_list, [self.variables_flat], variables_name, ['flat'])
        self.unpack_variables_fn = casadi.Function('unpack_variables_fn', [self.variables_flat], variables_list, ['flat'], variables_name)
        ## Box constraints
        lower_bounds = self.unpack_variables_fn(flat=-float('inf'))
        upper_bounds = self.unpack_variables_fn(flat=float('inf'))
        #lower_bounds['inputs'][:,0] = -40.0 # Minimum force
        #upper_bounds['inputs'][:,0] = 40.0 # Maximum force
        ## Initial state
        lower_bounds['states'][0, self.index_x]           = x0[0,self.index_x]
        upper_bounds['states'][0, self.index_x]           = x0[0,self.index_x]
        lower_bounds['states'][0, self.index_x_dot]       = x0[0,self.index_x_dot]
        upper_bounds['states'][0, self.index_x_dot]       = x0[0,self.index_x_dot]
        lower_bounds['states'][0, self.index_theta]       = x0[0,self.index_theta]
        upper_bounds['states'][0, self.index_theta]       = x0[0,self.index_theta]
        lower_bounds['states'][0, self.index_theta_dot]   = x0[0,self.index_theta_dot]
        upper_bounds['states'][0, self.index_theta_dot]   = x0[0,self.index_theta_dot]
        ## Final state
        #lower_bounds['states'][-1,self.index_x]           = self.xf[self.index_x]
        #upper_bounds['states'][-1,self.index_x]           = self.xf[self.index_x]
        #lower_bounds['states'][-1,self.index_x_dot]       = self.xf[self.index_x_dot]
        #upper_bounds['states'][-1,self.index_x_dot]       = self.xf[self.index_x_dot]
        #lower_bounds['states'][-1,self.index_theta]       = self.xf[self.index_theta]
        #upper_bounds['states'][-1,self.index_theta]       = self.xf[self.index_theta]
        #lower_bounds['states'][-1,self.index_theta_dot]   = self.xf[self.index_theta_dot]
        #upper_bounds['states'][-1,self.index_theta_dot]   = self.xf[self.index_theta_dot]
        
        #self.states[0,:]=SX(x0)
        X0 = self.states[0:self.config.mpc_horizon-1,:]
        X1 = self.states[1:self.config.mpc_horizon,:]
        # RK4
        defect = self.step(X0,self.inputs) - X1
        defect = casadi.reshape(defect, -1, 1)
        ## Optimization objective
        #objective = casadi.sum1(casadi.sum2(self.inputs**2))
        #objective = casadi.sum1(casadi.sqrt(casadi.sum2((self.states-np.tile(self.xf,(self.config.mpc_horizon,1)))**2)))
        potential_eng, kinetic_eng = self.cal_energy()
        
        objective = casadi.sum1(kinetic_eng-potential_eng)
        opts = {'ipopt.print_level':0, 'print_time':False}
        solver = casadi.nlpsol('solver', 'ipopt', {'x':self.variables_flat, 'f':objective, 'g':defect},opts)
        
        result = solver(x0=self.init_guess, lbg=0.0, ubg=0.0,
                lbx=self.pack_variables_fn(**lower_bounds)['flat'],
                ubx=self.pack_variables_fn(**upper_bounds)['flat'])
        self.init_guess = result['x']
        results = self.unpack_variables_fn(flat=result['x'])

        #states = np.array(results['states'])
        #actions = np.array(results['inputs'])
        #a = np.hstack((states[:self.config.mpc_horizon-1],actions))
        #print(a.shape)
        #np.savetxt("./out/out.csv", a, delimiter=",", header='x,x_dot,theta,theta_dot,u')
#
        #visualize_obj = CartPoleVisualizer(np.array(results['states']), self.config)
        #visualize_obj.plot(np.array(results['states']),np.array(results['inputs']))
        #visualize_obj.gen_animation()
        return results['inputs'][0].__float__()

