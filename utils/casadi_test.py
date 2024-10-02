from casadi import *

config = Config()
N = int(config.T/config.dt) # number of control intervals
T = config.T   # Final time

opti = Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(4,N+1) # state trajectory
pos         = X[0,:]
speed       = X[1,:]
angle       = X[2,:]
angular_vel = X[3,:]
U = opti.variable(1,N)   # control trajectory (force)

V = U[1:N]**2
print(V.reshape().shape)
V = sum(V)
# ---- objective          ---------
opti.minimize(V) # race in minimal time

# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[1], 
                        (1/(config.m_cart+config.m_pole*sin(theta)**2))*(u+config.m_pole*sin(theta)*(config.l*x[3]**2+config.g*cos(theta))), 
                        x[3],
                        (1/(config.l*(config.m_cart+config.m_pole*sin(theta)**2)))*(-u*cos(theta)-config.m_pole*config.l*x[3]**2*cos(theta)*sin(theta)-(config.m_cart+config.m_pole)*config.g*sin(theta))) # dx/dt = f(x,u)

for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+config.dt/2*k1, U[:,k])
   k3 = f(X[:,k]+config.dt/2*k2, U[:,k])
   k4 = f(X[:,k]+config.dt*k3,   U[:,k])
   x_next = X[:,k] + config.dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
opti.subject_to(opti.bounded(0,U,20)) # control is limited

# ---- boundary conditions --------
opti.subject_to(pos[0]==0)   
opti.subject_to(speed[0]==0)  
opti.subject_to(angle[0]==pi+0.1)  
opti.subject_to(angular_vel[0]==0)  
opti.subject_to(angle[-1]==pi)  
opti.subject_to(angular_vel[-1]==0)
opti.subject_to(speed[-1]==0)

# ---- misc. constraints  ----------
opti.subject_to(T>=0) # Time must be positive

# ---- initial values for solver ---
opti.set_initial(speed, 1)
opti.set_initial(T, 1)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

# ---- post-processing        ------
from pylab import plot, step, figure, legend, show, spy

plot(sol.value(speed),label="speed")
plot(sol.value(pos),label="pos")
plot(sol.value(angle),label="angle")
plot(sol.value(angular_vel),label="angular vel")
step(range(N),sol.value(U),'k',label="force")
legend(loc="upper left")

#figure()
#spy(sol.value(jacobian(opti.g,opti.x)))
#figure()
#spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

show()
exit(0)
