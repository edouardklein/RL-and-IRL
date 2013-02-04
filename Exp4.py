# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Mountain Car
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

ACTION_SPACE=[-1,0,1]

def mountain_car_next_step(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    next_speed = next_speed if next_speed > -0.07 else -0.07
    next_speed = next_speed if next_speed < 0.07 else 0.07
    next_position = next_position if next_position > -1.2 else -1.2
    next_position = next_position if next_position < 0.6 else 0.6
    return array([next_position,next_speed])

# <codecell>

def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])

mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,5),linspace(-0.07,0.07,5))

mountain_car_sigma_position = 2*pow((0.6+1.2)/5.,2)
mountain_car_sigma_speed = 2*pow((0.07+0.07)/5.,2)

def mountain_car_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(5*5,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position 
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    return array(psi).reshape((5*5,1))

def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros((5*5*3,1))
    answer[index_action*5*5:index_action*5*5+5*5] = mountain_car_psi(state)
    return answer

mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(75,1))

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else -1

def mountain_car_training_data():
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,100):
        state = mountain_car_uniform_state()
        for i in range(0,50):
            action = random_policy(state)
            next_state = mountain_car_next_step(state, action)
            reward = mountain_car_reward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)


    
data = mountain_car_training_data()
policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )

# <codecell>

def mountain_car_testing_data(policy):
    traj = []
    state = array([0.3,0])
    t=0
    reward = -1
    while t < 300 and reward == -1:
        t+=1
        action = policy(state)
        next_state = mountain_car_next_step(state, action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, reward]))
        state=next_state
    return array(traj)


state = array([0.5,0])
action = policy(state)
action
next_state = mountain_car_next_step(state, action)
next_state
data_test = mountain_car_testing_data(policy)
#data_test

# <codecell>

def mountain_car_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-1.2,0.6,30)
    speed = linspace(-0.07,0.07,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    #fig = figure()
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1))
        colorbar()
def mountain_car_plot_policy( policy ):
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    mountain_car_plot(two_args_pol,contour_levels=3)

def mountain_car_V(omega):
    policy = greedy_policy( omega, mountain_car_phi, ACTION_SPACE )
    def V(pos,speed):
        actions = policy(zip_stack(pos,speed))
        Phi=mountain_car_phi(zip_stack(pos,speed,actions))
        return squeeze(dot(omega.transpose(),Phi))
    return V


mountain_car_plot(mountain_car_V(omega))
scatter(data_test[:,0],data_test[:,1])
#scatter(data[:,0],data[:,1])
figure()
mountain_car_plot_policy(policy)
data_test.shape

# <codecell>

data_test.shape

