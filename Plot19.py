# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Des graphes sur le mountain car pour le manuscrit
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#Code from Exp5
#Mountain Car
from stuff import *
from pylab import *
from random import *
import pickle
import numpy
from rl import *
import sys

NB_SAMPLES=100
#NB_SAMPLES=int(sys.argv[1])
RAND_STRING=str(int(rand()*10000000000))


ACTION_SPACE=[-1,0,1]



def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    if not -0.07 <= next_speed <= 0.07:
        next_speed = sign(next_speed)*0.07
    if not -1.2 <= next_position <= 0.6:
        next_speed=0.
        next_position = -1.2 if next_position < -1.2 else 0.6
    return array([next_position,next_speed])

def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])

mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,7),linspace(-0.07,0.07,7))

mountain_car_sigma_position = 2*pow((0.6+1.2)/10.,2)
mountain_car_sigma_speed = 2*pow((0.07+0.07)/10.,2)

def mountain_car_single_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(7*7,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position 
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    psi.append(1.)
    return array(psi).reshape((7*7+1,1))

mountain_car_psi= non_scalar_vectorize(mountain_car_single_psi,(2,),(50,1))

def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros(((7*7+1)*3,1))
    answer[index_action*(7*7+1):index_action*(7*7+1)+7*7+1] = mountain_car_psi(state)
    return answer

mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(150,1))

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0

def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 300 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))


def mountain_car_training_data(freward=mountain_car_reward,traj_length=5,nb_traj=1000):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,nb_traj):
        state = mountain_car_uniform_state()
        reward=0
        t=0
        while t < traj_length and reward == 0:
            t+=1
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)

def mountain_car_manual_policy(state):
    position,speed = state
    return -1. if speed <=0 else 1.

def mountain_car_interesting_state():
    position = numpy.random.uniform(low=-1.2,high=-0.9)
    speed = numpy.random.uniform(low=-0.07,high=0)
    return array([position,speed])

def mountain_car_IRL_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = mountain_car_manual_policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=next_state
    return array(traj)

def mountain_car_IRL_data(nbsamples):
    data = mountain_car_IRL_traj()
    while len(data) < nbsamples:
        data = vstack([data,mountain_car_IRL_traj()])
    return data[:nbsamples]

#From Exp4
my_cmap = matplotlib.pyplot.cm.cubehelix
def mountain_car_plot( f, draw_contour=True, contour_levels=50, draw_surface=False ):
    '''Display a surface plot of function f over the state space'''
    pos = linspace(-1.2,0.6,30)
    speed = linspace(-0.07,0.07,30)
    pos,speed = meshgrid(pos,speed)
    Z = f(pos,speed)
    fig = figure(figsize=(1,1))
    if draw_surface:
        ax=Axes3D(fig)
        ax.plot_surface(pos,speed,Z)
    if draw_contour:
        axis('off')
        contourf(pos,speed,Z,levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),contour_levels+1),
                    cmap=my_cmap)
        
        #colorbar()
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

def mountain_car_theta_psi(theta):
    def V(pos,speed):
        Psi=mountain_car_psi(zip_stack(pos,speed))
        print((Psi.shape,theta.transpose().shape))
        return squeeze(dot(theta.transpose(),Psi))
    return V
theta = zeros((7*7+1,1))
theta[3+2*7] = 1

mountain_car_plot(mountain_car_theta_psi(theta))

# <codecell>

#Illustrer les gaussiennes
for i in range(0,7):
    for j in range(0,7):
        theta = zeros((7*7+1,1))
        theta[j+i*7] = 1
        mountain_car_plot(mountain_car_theta_psi(theta))
        savefig("Mountain_car_psi_{i}x{j}.pdf".format(i=i,j=j),bbox_inches='tight')

# <codecell>

k = 0
for states in zip_stack(mountain_car_mu_position,mountain_car_mu_speed):
    for s in states:
        data_MC = []
        reward = 0
        state = s
        while reward == 0:
            action = mountain_car_manual_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = mountain_car_reward(hstack([state, action, next_state]))
            data_MC.append(hstack([state, action, next_state, reward]))
            state=next_state
        data_MC = array(data_MC)
        GAMMAS = range(0,len(data_MC))
        GAMMAS = array([el for el in map( lambda x: pow(GAMMA,x), GAMMAS)])
        state_action = data_MC[0,:3]
        state = data_MC[0,:2]
        action = data_MC[0,2]
        mu = None
        if len(data_MC) > 1:
            mu = dot( GAMMAS,squeeze(mountain_car_psi(data_MC[:,:2])))
        else:
            mu = squeeze(mountain_car_psi(squeeze(data_MC[:,:2])))
        mountain_car_plot(mountain_car_theta_psi(mu))
        #plot(s[0],s[1],marker='o',color='pink')
        savefig("Mountain_car_mu_{k}.pdf".format(k=k),bbox_inches='tight')
        k+=1

# <codecell>

pos = linspace(-1.2,0.6,30)
speed = linspace(-0.07,0.07,30)
pos,speed = meshgrid(pos,speed)
states=zip_stack(pos,speed)
dic_mu_E = {}
for s in states.reshape(-1,2):
    data_MC = []
    reward = 0
    state = s
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        data_MC.append(hstack([state, action, next_state, reward]))
        state=next_state
    data_MC = array(data_MC)
    GAMMAS = range(0,len(data_MC))
    GAMMAS = array([el for el in map( lambda x: pow(GAMMA,x), GAMMAS)])
    state_action = data_MC[0,:3]
    state = data_MC[0,:2]
    action = data_MC[0,2]
    mu = None
    if len(data_MC) > 1:
        mu = dot( GAMMAS,squeeze(mountain_car_psi(data_MC[:,:2])))
    else:
        mu = squeeze(mountain_car_psi(squeeze(data_MC[:,:2])))
    dic_mu_E[str(s)] = mu

# <codecell>

dic_mu_E
f_mu = lambda s:dic_mu_E[str(s)]
vf_mu = non_scalar_vectorize(f_mu,(2,),(50,1)) 
Z = squeeze(vf_mu(states.reshape(-1,2)))
for i in range(0,7):
    for j in range(0,7):
        k = j+i*7
        figure(figsize=(1,1))
        axis('off')
        contourf(pos,speed,Z[:,k].reshape(pos.shape),levels=linspace(min(Z.reshape(-1)),max(Z.reshape(-1)),51),
                    cmap=my_cmap)
        savefig("Mountain_car_mu_{i}x{j}.pdf".format(i=i,j=j),bbox_inches='tight')
    

# <codecell>

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0
def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    if not -0.07 <= next_speed <= 0.07:
        next_speed = sign(next_speed)*0.07
    if not -1.2 <= next_position <= 0.6:
        next_speed=0.
        next_position = -1.2 if next_position < -1.2 else 0.6
    return array([next_position,next_speed])

def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 300 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))
def mountain_car_manual_policy(state):
    position,speed = state
    return -1. if speed <=0 else 1.
plottable_episode_length = mountain_car_episode_vlength(mountain_car_manual_policy)
X = linspace(-1.2,0.6,30)
Y = linspace(-0.07,0.07,30)
X,Y = meshgrid(X,Y)
Z6 = plottable_episode_length(X,Y)

# <codecell>

fig = figure()
majorLocator   = MultipleLocator(.5)
majorFormatter = FormatStrFormatter('$%.1f$')
minorLocator   = MultipleLocator(.1)
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt = contourf(X,Y,Z6,50,cmap=my_cmap)
cb = colorbar()
cb.ax.yaxis.set_ticks_position('left')
plt.ax.yaxis.set_ticks_position('right')
plt.ax.xaxis.set_major_locator(majorLocator)
plt.ax.xaxis.set_major_formatter(majorFormatter)
plt.ax.xaxis.set_minor_locator(minorLocator)
cb.ax.set_position([0.07, 0.12, .5, .78])
scatter(TRAJS[:,0],TRAJS[:,1],c=TRAJS[:,2])
savefig("Mountain_car_Expert_traj_length.pdf")

# <codecell>

def mountain_car_IRL_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = mountain_car_manual_policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=next_state
    return array(traj)
TRAJS = mountain_car_IRL_traj()

# <codecell>

fig=figure(figsize=(5,5))
plt = contourf(X,Y,Z6,50,cmap=my_cmap)
plt.ax.yaxis.set_ticks_position('right')
plt.ax.xaxis.set_major_locator(majorLocator)
plt.ax.xaxis.set_major_formatter(majorFormatter)
plt.ax.xaxis.set_minor_locator(minorLocator)
scatter(TRAJS[:,0],TRAJS[:,1],c=TRAJS[:,2])
axis([-1.2,0.6,-0.07,0.07])
savefig("Mountain_car_Expert_traj_example.pdf")

