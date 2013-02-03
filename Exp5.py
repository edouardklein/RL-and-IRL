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
NB_SAMPLES=10
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
    return 1 if position > 0.5 else 0

def mountain_car_training_data(freward=mountain_car_reward):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,500):
        state = mountain_car_uniform_state()
        for i in range(0,10):
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)
omega=genfromtxt("mountain_car_expert_omega.mat")
policy=greedy_policy(omega, mountain_car_phi, ACTION_SPACE)
def mountain_car_testing_data(policy):
    traj = []
    state = mountain_car_interesting_state()
    t=0
    reward = 0
    while t < 300 and reward == 0:
        t+=1
        action = policy(state)
        next_state = mountain_car_next_state(state, action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, reward]))
        state=mountain_car_interesting_state()
    return array(traj)
while True:#Do..while
    data = vstack([mountain_car_testing_data(policy) for i in range(0,10)])
    randomly_selected_trans = array([choice(data) for i in range(0,NB_SAMPLES)])
    actions = randomly_selected_trans[:,2]
    if any(actions == 0) and any(actions == -1) and any(actions == 1):
        break
psi=mountain_car_psi
phi=mountain_car_phi
traj=randomly_selected_trans
s=traj[:,:2]
a=traj[:,2]
#Classification

from sklearn import svm
clf = svm.SVC(C=10, probability=True)
clf.fit(s, a)
import pickle
RAND_STRING=str(int(rand()*100000))
with open('data/Classif_'+str(NB_SAMPLES)+"_"+RAND_STRING+".obj", 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
#Données pour la regression
column_shape = (len(traj),1)
s = traj[:,0:2]
a = traj[:,2].reshape(column_shape)
sa = traj[:,0:3]
s_dash = traj[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
##Avec l'heuristique : 
regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in ACTION_SPACE] 
def add_output_column( reg_mat ):
    actions = reg_mat[:,-1].reshape(column_shape)
    hat_r_bool_table = array(actions==a)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
    return hstack([reg_mat,output_column])
regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#Régression
from sklearn.svm import SVR
y = regression_matrix[:,-1]
X = regression_matrix[:,:-1]
reg = SVR(C=1.0, epsilon=0.2)
reg.fit(X, y)
CSI_reward = lambda sas:reg.predict(sas[:3])[0]
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
#Evaluation de l'IRL
data = mountain_car_training_data(freward=CSI_reward)
policy_CSI,omega_CSI = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )
savetxt("data/CSI_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_CSI)
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
    return mountain_car_episode_length(-0.9,0,policy)

print "Nb_samples :\t"+str(NB_SAMPLES)
print "Length CSI :\t"+str(mountain_car_tricky_episode_length(policy_CSI))
print "Length Class :\t"+str(mountain_car_tricky_episode_length(pi_c))
print "Length exp :\t"+str(mountain_car_tricky_episode_length(policy))

