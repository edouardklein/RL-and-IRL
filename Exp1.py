# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Misc.

# <codecell>

#!/usr/bin/env python
from random import choice
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D

def non_scalar_vectorize(func, input_shape, output_shape):
    """Return a featurized version of func, where func takes a potentially matricial argument and returns a potentially matricial answer.

    These functions can not be naively vectorized by numpy's vectorize.
 
    With vfunc = non_scalar_vectorize( func, (2,), (10,1) ),
    
    func([p,s]) will return a 2D matrix of shape (10,1).

    func([[p1,s1],...,[pn,sn]]) will return a 3D matrix of shape (n,10,1).

    And so on.
    """
    def vectorized_func(arg):
        #print 'Vectorized : arg = '+str(arg)
        nbinputs = prod(arg.shape)/prod(input_shape)
        if nbinputs == 1:
            return func(arg)
        outer_shape = arg.shape[:len(arg.shape)-len(input_shape)]
        outer_shape = outer_shape if outer_shape else (1,)
        arg = arg.reshape((nbinputs,)+input_shape)
        answers=[]
        for input_matrix in arg:
            answers.append(func(input_matrix))
        return array(answers).reshape(outer_shape+output_shape)
    return vectorized_func

def zip_stack(*args):
    """Given matrices of same shape, return a matrix whose elements are tuples from the arguments (i.e. with one more dimension).

    zip_stacking three matrices of shape (n,p) will yeld a matrix of shape (n,p,3)
    """
    shape = args[0].shape
    nargs = len(args)
    args = [m.reshape(-1) for m in args]
    return array(zip(*args)).reshape(shape+(nargs,))
#zip_stack(array([[1,2,3],[4,5,6]]),rand(2,3))

# <headingcell level=2>

# Reinforcement Learning Code

# <codecell>

def argmax( set, func ):
     return max( zip( set, map(func,set) ), key=lambda x:x[1] )[0]

def greedy_policy( omega, phi, A ): 
    def policy( *args ):
        state_actions = [hstack(args+(a,)) for a in A]
        q_value = lambda sa: float(dot(omega.transpose(),phi(sa)))
        best_action = argmax( state_actions, q_value )[-1] #FIXME6: does not work for multi dimensional actions
        return best_action
    vpolicy = non_scalar_vectorize( policy, (2,), (1,1) )
    return lambda state: vpolicy(state).reshape(state.shape[:-1]+(1,))

#test_omega=zeros((30,1))
#test_omega[1]=1.
#pol = greedy_policy( test_omega, inverted_pendulum_phi, ACTION_SPACE )
#[pol(rand(2)),pol(rand(3,2)).shape,pol(rand(3,3,2)).shape]

# <codecell>

def lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=1):
    #print "shapes of phi de sa, phi de sprim a prim, rewards"+str(phi_sa.shape)+str(phi_sa_dash.shape)+str(rewards.shape)
    A = zeros((phi_dim,phi_dim))
    b = zeros((phi_dim,1))
    for phi_t,phi_t_dash,reward in zip(phi_sa,phi_sa_dash,rewards):
        A = A + dot( phi_t,
                     (phi_t - GAMMA*phi_t_dash).transpose())
        b = b + phi_t*reward
    return dot(inv(A + LAMBDA*identity( phi_dim )),b)

def lspi( data, s_dim=1, a_dim=1, A = [0], phi=None, phi_dim=1, epsilon=0.01, iterations_max=30,
          plot_func=None):
    nb_iterations=0
    sa = data[:,0:s_dim+a_dim]
    phi_sa = phi(sa)
    s_dash = data[:,s_dim+a_dim:s_dim+a_dim+s_dim]
    rewards = data[:,s_dim+a_dim+s_dim]
    omega = zeros(( phi_dim, 1 ))
    #omega = genfromtxt("../Code/InvertedPendulum/omega_E.mat").reshape(30,1)
    diff = float("inf")
    cont = True
    policy = greedy_policy( omega, phi, A )
    while cont:
        if plot_func:
            plot_func(omega)
        sa_dash = hstack([s_dash,policy(s_dash)])
        phi_sa_dash = phi(sa_dash)
        omega_dash = lstdq(phi_sa, phi_sa_dash, rewards, phi_dim=phi_dim)
        diff = norm( omega_dash-omega )
        omega = omega_dash
        policy = greedy_policy( omega, phi, A )
        nb_iterations+=1
        print "LSPI, iter :"+str(nb_iterations)+", diff : "+str(diff)
        if nb_iterations > iterations_max or diff < epsilon:
            cont = False
    return policy,omega

# <headingcell level=2>

# Inverted Pendulum-specific code

# <codecell>

RANDOM_RUN_LENGTH=10000
EXPERT_RUN_LENGTH=3000
TRANS_WIDTH=6
ACTION_SPACE=[0,1,2]
GAMMA=0.9 #Discout factor
LAMBDA=0.1 #Regularization coeff for LSTDQ

# <codecell>

def inverted_pendulum_single_psi( state ):
    position,speed=state
    answer = zeros((10,1))
    index = 0
    answer[index] = 1.
    index+=1
    for i in linspace(-pi/4,pi/4,3):
        for j in linspace(-1,1,3):
            answer[index] = exp(-(pow(position-i,2) +
                                  pow(speed-j,2))/2.)
            index+=1
    #print "psi stops ar index "+str(index)
    return answer

inverted_pendulum_psi = non_scalar_vectorize( inverted_pendulum_single_psi,(2,), (10,1) )

#[inverted_pendulum_psi(rand(2)).shape,
# inverted_pendulum_psi(rand(2,2)).shape,
# inverted_pendulum_psi(rand(3,5,2)).shape]

# <codecell>

def inverted_pendulum_single_phi(state_action):
    position, speed, action = state_action
    answer = zeros((30,1))
    index = action*10
    answer[ index:index+10 ] = inverted_pendulum_single_psi( [position, speed] )
    return answer

inverted_pendulum_phi = non_scalar_vectorize(inverted_pendulum_single_phi, (3,), (30,1))

#[inverted_pendulum_phi(rand(3)).shape,
# inverted_pendulum_phi(rand(3,3)).shape,
# inverted_pendulum_phi(rand(4,5,3)).shape]

# <codecell>

def inverted_pendulum_V(omega):
    policy = greedy_policy( omega, inverted_pendulum_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=inverted_pendulum_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V

# <codecell>

def inverted_pendulum_next_state(state, action):
    position,speed = state
    noise = rand()*20.-10.
    control = None
    if action == 0:
        control = -50 + noise;
    elif action == 1:
        control = 0 + noise;
    else: #action==2
        control = 50 + noise;
    g = 9.8;
    m = 2.0;
    M = 8.0;
    l = 0.50;
    alpha = 1./(m+M);
    step = 0.1;
    acceleration = (g*sin(position) - 
                    alpha*m*l*pow(speed,2)*sin(2*position)/2. - 
                    alpha*cos(position)*control) / (4.*l/3. - alpha*m*l*pow(cos(position),2))
    next_position = position +speed*step;
    next_speed = speed + acceleration*step;
    return array([next_position,next_speed])

def inverted_pendulum_reward( state ):
    position,speed = state
    if abs(position)>pi/2.:
        return -1.
    return 0.

def inverted_pendulum_uniform_initial_state():
    return (rand(2)*2.-1.)*pi/2.

def inverted_pendulum_optimal_initial_state():
    return rand(2)*0.2-0.1

def inverted_pendulum_trace( pi,run_length=RANDOM_RUN_LENGTH,
                             initial_state=inverted_pendulum_optimal_initial_state ):
    data = zeros((run_length, TRANS_WIDTH))
    state = initial_state()
    for i,void in enumerate( data ):
        action = pi( state )
        new_state = inverted_pendulum_next_state( state, action )
        reward = inverted_pendulum_reward( new_state )
        data[i,:] = hstack([state,action,new_state,[reward]])
        if reward == 0.:
            state = new_state
        else: #Pendulum has fallen
            state = initial_state()
    return data

def inverted_pendulum_random_trace():
    pi = lambda s: choice(ACTION_SPACE)
    return inverted_pendulum_trace( pi )

def inverted_pendulum_expert_trace():
    data = inverted_pendulum_random_trace()
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30 )
    inverted_pendulum_plot(inverted_pendulum_V(omega))
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
    return inverted_pendulum_trace( policy,run_length=EXPERT_RUN_LENGTH ),policy

# <codecell>

def inverted_pendulum_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-pi,pi,30)
    speed = linspace(-pi,pi,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    fig = figure()
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1))
        colorbar()
    show()
#test_omega=zeros((30,1))
#test_omega[5]=1.
#inverted_pendulum_plot(inverted_pendulum_V(test_omega))

# <headingcell level=2>

# Experiment 1 Code

# <codecell>

            
data_random = inverted_pendulum_random_trace()
data_expert,policy = inverted_pendulum_expert_trace()
random_falls_rate = - mean( data_random[:,5] )
expert_falls_rate = - mean( data_expert[:,5] )
print "Rate of falls for random controller "+str(random_falls_rate)
print "Rate of falls for expert controller "+str(expert_falls_rate)
#reward = cascading_irl( data_expert )
#value_function = lspi( reward, data_random )
#inverted_pendulum_plot( reward, "Reward.pdf" )
#inverted_pendulum_plot( value_function, "ValueFunction.pdf" )
#sum(data_random[:,5])
#plot(data_random[:,0],data_random[:,1],ls='',marker='o')

