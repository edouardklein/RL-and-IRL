# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Mountain Car
import matplotlib
matplotlib.use('Agg')
X=[10]
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

def mountain_car_testing_state():
    position = numpy.random.uniform(low=-1.2,high=0.5)
    speed = numpy.random.uniform(low=-0.07,high=0.07)
    return array([position,speed])

def mountain_car_mean_performance(policy):
    return mean([mountain_car_episode_length(state[0],state[1],policy) for state in [mountain_car_testing_state() for i in range(0,10)]])

# <codecell>

import glob
import pickle

#X=[10, 100, 300]
Y_mean_CSI=[]
Y_deviation_CSI=[]
Y_min_CSI=[]
Y_max_CSI=[]
Y_mean_Class=[]
Y_deviation_Class=[]
Y_min_Class=[]
Y_max_Class=[]
Y_mean_SCIRL=[]
Y_deviation_SCIRL=[]
Y_min_SCIRL=[]
Y_max_SCIRL=[]
Y_mean_SCIRLMC=[]
Y_deviation_SCIRLMC=[]
Y_min_SCIRLMC=[]
Y_max_SCIRLMC=[]
Y_mean_Expert=[]
Y_deviation_Expert=[]
Y_min_Expert=[]
Y_max_Expert=[]
Y_mean_RE=[]
Y_deviation_RE=[]
Y_min_RE=[]
Y_max_RE=[]

all_data_CSI=[]
all_data_Classif=[]
all_data_Expert=[]
all_data_SCIRL=[]
all_data_SCIRLMC=[]
all_data_RE=[]

for x in X:
    CSI_files = glob.glob("data/CSI_omega_"+str(x)+"_*.mat")#
    Classif_files = glob.glob("data/Classif_"+str(x)+"_*.obj")#
    #Expert_files = glob.glob("data/Expert_omega_"+str(x)+"_*.mat")##
    SCIRL_files = glob.glob("data/SCIRL_omega_"+str(x)+"_*.mat")#
    SCIRLMC_files = glob.glob("data/SCIRLMC_omega_"+str(x)+"_*.mat")#
    RE_files = glob.glob("data/RE_omega_"+str(x)+"_*.mat")#
    #print CSI_files
    #print Classif_files
    #print Expert_files
    #print SCIRL_files

    data_CSI=[]
    data_Classif=[]
    data_Expert=[]
    data_SCIRL=[]
    data_SCIRLMC=[]
    data_RE=[]

    for CSI_file,Classif_file,SCIRL_file,SCIRLMC_file,RE_file in zip(CSI_files,Classif_files,SCIRL_files,SCIRLMC_files,RE_files):
        omega_CSI=genfromtxt(CSI_file)
        policy_CSI=greedy_policy(omega_CSI, mountain_car_phi, ACTION_SPACE)
        perf_CSI = mountain_car_mean_performance(policy_CSI)
        data_CSI.append(perf_CSI) 
        
        with open(Classif_file, 'rb') as input:
            clf = pickle.load(input)
        clf_predict= lambda state : clf.predict(squeeze(state))
        vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
        pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
        perf_Classif = mountain_car_mean_performance(pi_c)
        data_Classif.append(perf_Classif) 
        
        #omega=genfromtxt(Expert_file)
        #policy=greedy_policy(omega, mountain_car_phi, ACTION_SPACE)
        perf_Expert = mountain_car_mean_performance(mountain_car_manual_policy)
        data_Expert.append(perf_Expert)
        
        omega_SCIRL=genfromtxt(SCIRL_file)
        policy_SCIRL=greedy_policy(omega_SCIRL, mountain_car_phi, ACTION_SPACE)
        perf_SCIRL = mountain_car_mean_performance(policy_SCIRL)
        data_SCIRL.append(perf_SCIRL)
        
        omega_SCIRLMC=genfromtxt(SCIRLMC_file)
        policy_SCIRLMC=greedy_policy(omega_SCIRLMC, mountain_car_phi, ACTION_SPACE)
        perf_SCIRLMC = mountain_car_mean_performance(policy_SCIRLMC)
        data_SCIRLMC.append(perf_SCIRLMC)
        
        omega_RE=genfromtxt(RE_file)
        policy_RE=greedy_policy(omega_RE, mountain_car_phi, ACTION_SPACE)
        perf_RE = mountain_car_mean_performance(policy_RE)
        data_RE.append(perf_RE)
        
        
        print "\tNb_samples :\t"+str(x)
        print "\tLength CSI :\t"+str(perf_CSI)
        print "\tLength Class :\t"+str(perf_Classif)
        print "\tLength SCIRL :\t"+str(perf_SCIRL)
        print "\tLength SCIRLMC :\t"+str(perf_SCIRLMC)
        print "\tLength RE :\t"+str(perf_RE)
        print "\tLength exp :\t"+str(perf_Expert)
    print "Nb_samples :\t"+str(x)
    print "Mean length CSI :\t"+str(mean(data_CSI))
    print "Mean length Class :\t"+str(mean(data_Classif))
    print "Mean length Expert :\t"+str(mean(data_Expert))
    print "Mean length SCIRL :\t"+str(mean(data_SCIRL))
    print "Mean length RE :\t"+str(perf_RE)
    print "Mean length SCIRLMC :\t"+str(mean(data_SCIRLMC))
    
    all_data_CSI.append(data_CSI)
    all_data_Classif.append(data_Classif)
    all_data_Expert.append(data_Expert)
    all_data_SCIRL.append(data_SCIRL)
    all_data_SCIRLMC.append(data_SCIRLMC)
    all_data_RE.append(data_RE)
    
    Y_mean_CSI.append(mean(data_CSI))
    Y_deviation_CSI.append(sqrt(var(data_CSI)))
    Y_min_CSI.append(min(data_CSI))
    Y_max_CSI.append(max(data_CSI))
    Y_mean_Class.append(mean(data_Classif))
    Y_deviation_Class.append(sqrt(var(data_Classif)))
    Y_min_Class.append(min(data_Classif))
    Y_max_Class.append(max(data_Classif))
    Y_mean_SCIRL.append(mean(data_SCIRL))
    Y_deviation_SCIRL.append(sqrt(var(data_SCIRL)))
    Y_min_SCIRL.append(min(data_SCIRL))
    Y_max_SCIRL.append(max(data_SCIRL))
    Y_mean_SCIRLMC.append(mean(data_SCIRLMC))
    Y_deviation_SCIRLMC.append(sqrt(var(data_SCIRLMC)))
    Y_min_SCIRLMC.append(min(data_SCIRLMC))
    Y_max_SCIRLMC.append(max(data_SCIRLMC))
    Y_mean_RE.append(mean(data_RE))
    Y_deviation_RE.append(sqrt(var(data_RE)))
    Y_min_RE.append(min(data_RE))
    Y_max_RE.append(max(data_RE))
    Y_mean_Expert.append(mean(data_Expert))
    Y_deviation_Expert.append(sqrt(var(data_Expert)))
    Y_min_Expert.append(min(data_Expert))
    Y_max_Expert.append(max(data_Expert))

Y_mean_CSI=array(Y_mean_CSI)
Y_deviation_CSI=array(Y_deviation_CSI)
Y_min_CSI=array(Y_min_CSI)
Y_max_CSI=array(Y_max_CSI)
Y_mean_Class=array(Y_mean_Class)
Y_deviation_Class=array(Y_deviation_Class)
Y_min_Class=array(Y_min_Class)
Y_max_Class=array(Y_max_Class)
Y_mean_SCIRL=array(Y_mean_SCIRL)
Y_deviation_SCIRL=array(Y_deviation_SCIRL)
Y_min_SCIRL=array(Y_min_SCIRL)
Y_max_SCIRL=array(Y_max_SCIRL)
Y_deviation_SCIRLMC=array(Y_deviation_SCIRLMC)
Y_min_SCIRLMC=array(Y_min_SCIRLMC)
Y_max_SCIRLMC=array(Y_max_SCIRLMC)
Y_mean_Expert=array(Y_mean_Expert)
Y_deviation_Expert=array(Y_deviation_Expert)
Y_min_Expert=array(Y_min_Expert)
Y_max_Expert=array(Y_max_Expert)
Y_mean_RE=array(Y_mean_RE)
Y_deviation_RE=array(Y_deviation_RE)
Y_min_RE=array(Y_min_RE)
Y_max_RE=array(Y_max_RE)
    

# <codecell>

#With N the number of runs and |X| the number of abcissas, we save the performance of the various algos in |X|xN matrices
Xstr = "_".join(map(str,X))
savetxt("data/CSI_X_"+Xstr+".mat", array(all_data_CSI))
savetxt("data/SCIRL_X_"+Xstr+".mat", array(all_data_SCIRL))
savetxt("data/SCIRLMC_X_"+Xstr+".mat", array(all_data_SCIRLMC))
savetxt("data/RE_X_"+Xstr+".mat", array(all_data_RE))
savetxt("data/Classif_X_"+Xstr+".mat", array(all_data_Classif))
savetxt("data/Expert_X_"+Xstr+".mat", array(all_data_Expert))

