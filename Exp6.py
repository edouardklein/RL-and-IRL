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
NB_SAMPLES=100
def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    next_speed = next_speed if next_speed > -0.07 else -0.07
    next_speed = next_speed if next_speed < 0.07 else 0.07
    next_position = next_position if next_position > -1.2 else -1.2
    next_position = next_position if next_position < 0.6 else 0.6
    return array([next_position,next_speed])
def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])
def mountain_car_interesting_state():
    position = choice([numpy.random.uniform(low=-1.2,high=-0.8),numpy.random.uniform(low=0,high=0.6)])
    speed = choice([numpy.random.uniform(low=-0.07,high=-0.04),numpy.random.uniform(low=0.04,high=0.07)])
    if position < 0 and speed < 0:
        speed=-speed
    return array([position,speed])

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0

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
omega=genfromtxt("mountain_car_expert_omega.mat")
policy=greedy_policy(omega, mountain_car_phi, ACTION_SPACE)

# <codecell>

starting_states = [array([numpy.random.uniform(low=-1.2,high=-0.8),numpy.random.uniform(low=-0.07,high=-0.04)]) for i in range(0,2)]

# <codecell>

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

def mountain_car_tricky_episode_length(policy):
    #return mountain_car_episode_length(-0.9,-0.04,policy)
    return mean([mountain_car_episode_length(state[0],state[1], policy) for state in starting_states])

# <codecell>

import glob
import pickle

ABCISSAS=[10,100]
for x in ABCISSAS:
    CSI_files = glob.glob("data/CSI_omega_"+str(x)+"_*.mat")
    Classif_files = glob.glob("data/Classif_"+str(x)+"_*.obj")
    print CSI_files
    print Classif_files

    data_CSI=[]
    data_Classif=[]
    data_Expert=[]

    for CSI_file,Classif_file in zip(CSI_files,Classif_files):
        omega_CSI=genfromtxt(CSI_file)
        with open(Classif_file, 'rb') as input:
            clf = pickle.load(input)
        clf_predict= lambda state : clf.predict(squeeze(state))
        vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
        pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
        policy_CSI=greedy_policy(omega_CSI, mountain_car_phi, ACTION_SPACE)
        perf_CSI = mountain_car_tricky_episode_length(policy_CSI)
        perf_Classif = mountain_car_tricky_episode_length(pi_c)
        perf_Expert = mountain_car_tricky_episode_length(policy)
        data_CSI.append(perf_CSI) 
        data_Classif.append(perf_Classif) 
        data_Expert.append(perf_Expert) 
        #print "Nb_samples :\t"+str(x)
        #print "Length CSI :\t"+str(perf_CSI)
        #print "Length Class :\t"+str(perf_Classif)
        #print "Length exp :\t"+str(perf_Expert)
    print "Nb_samples :\t"+str(x)
    print "Mean length CSI :\t"+str(mean(data_CSI))
    print "Mean length Class :\t"+str(mean(data_Classif))
    print "Mean length exp :\t"+str(mean(data_Expert))

