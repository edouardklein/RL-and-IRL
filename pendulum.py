from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

RANDOM_RUN_LENGTH=5000
EXPERT_RUN_LENGTH=3000
TRANS_WIDTH=6
ACTION_SPACE=[0,1,2]


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


def inverted_pendulum_single_phi(state_action):
    position, speed, action = state_action
    answer = zeros((30,1))
    index = action*10
    answer[ index:index+10 ] = inverted_pendulum_single_psi( [position, speed] )
    return answer

inverted_pendulum_phi = non_scalar_vectorize(inverted_pendulum_single_phi, (3,), (30,1))


def inverted_pendulum_V(omega):
    policy = greedy_policy( omega, inverted_pendulum_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=inverted_pendulum_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V

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

def inverted_pendulum_single_reward( sas ):
    position,speed = sas[-2:]
    #print "position is "+str(position)
    if abs(position)>pi/2.:
    #    print "-1"
        return -1.
    #print "0"
    return 0.

inverted_pendulum_vreward = non_scalar_vectorize( inverted_pendulum_single_reward, (5,),(1,1) )
inverted_pendulum_reward = lambda sas:squeeze(inverted_pendulum_vreward(sas))

def inverted_pendulum_uniform_initial_state():
    return array(numpy.random.uniform(low=-pi/2, high=pi/2, size=2))

def inverted_pendulum_nice_initial_state():
    return array(numpy.random.uniform(low=-0.1, high=0.1, size=2))

def inverted_pendulum_trace( policy,run_length=RANDOM_RUN_LENGTH,
                             initial_state=inverted_pendulum_uniform_initial_state,
                             reward = inverted_pendulum_reward):
    data = zeros((run_length, TRANS_WIDTH))
    state = initial_state()
    for i,void in enumerate( data ):
        action = policy( state )
        new_state = inverted_pendulum_next_state( state, action )
        r = reward( hstack([state,action,new_state]) )
        data[i,:] = hstack([state,action,new_state,[r]])
        if r == 0.:
            state = new_state
        else: #Pendulum has fallen
            state = initial_state()
    return data

def inverted_pendulum_random_trace(reward=inverted_pendulum_reward,
                                    initial_state=inverted_pendulum_nice_initial_state):
    pi = lambda s: choice(ACTION_SPACE)
    return inverted_pendulum_trace( pi,reward=reward, initial_state=initial_state)

def inverted_pendulum_expert_trace( reward ):
    data = inverted_pendulum_random_trace(reward=reward)
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=10 )
    inverted_pendulum_plot(inverted_pendulum_V(omega))
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
    return inverted_pendulum_trace( policy,run_length=EXPERT_RUN_LENGTH ),policy,omega

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
    #show()
def inverted_pendulum_plot_policy( policy ):
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)

def inverted_pendulum_plot_SAReward(reward,policy):
    X = linspace(-pi,pi,30)
    Y = X
    X,Y = meshgrid(X,Y)
    XY = zip_stack(X,Y)
    XYA = zip_stack(X,Y,squeeze(policy(XY)))
    Z = squeeze(reward(XYA))
    contourf(X,Y,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),51))
    colorbar()

    
def inverted_pendulum_plot_SReward(reward,policy):
    X = linspace(-pi,pi,30)
    Y = X
    X,Y = meshgrid(X,Y)
    XY = zip_stack(X,Y)
    XYA = zip_stack(X,Y,squeeze(policy(XY)))
    Z = squeeze(reward(XYA))
    contourf(X,Y,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),51))
    colorbar()


inverted_pendulum_expert_omega = genfromtxt("inverted_pendulum_expert_omega.mat")

def inverted_pendulum_expert_policy():
    return greedy_policy(inverted_pendulum_expert_omega, inverted_pendulum_phi, ACTION_SPACE)

