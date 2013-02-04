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

def mountain_car_next_state(state,action):
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
def mountain_car_interesting_state():
    position = choice([numpy.random.uniform(low=-1.2,high=-0.8),numpy.random.uniform(low=0,high=0.6)])
    speed = choice([numpy.random.uniform(low=-0.07,high=-0.04),numpy.random.uniform(low=0.04,high=0.07)])
    return array([position,speed])

# <codecell>

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
    for i in range(0,100):
        state = mountain_car_uniform_state()
        for i in range(0,5):
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)

# <codecell>

    
data = mountain_car_training_data()
policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )

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
    return mountain_car_episode_length(-0.9,-0.04,policy)

# <codecell>

savetxt("./mountain_car_expert_omega.mat",omega)
while mountain_car_tricky_episode_length(policy) > 52:
    print "Expert still not good enough :\t"+str(mountain_car_tricky_episode_length(policy))
    data = mountain_car_training_data()
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )

# <codecell>

print "Expert good enough :\t"+str(mountain_car_tricky_episode_length(policy))
savetxt("./mountain_car_expert_omega.mat",omega)

# <codecell>

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
        state=next_state
    return array(traj)

state = array([0.5,0])
action = policy(state)
action
next_state = mountain_car_next_state(state, action)
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
figure()
mountain_car_plot_policy(policy)
data_test.shape

# <codecell>

data = vstack([mountain_car_testing_data(policy) for i in range(0,10)])

# <codecell>

randomly_selected_trans = array([choice(data) for i in range(0,10)])
mountain_car_plot(mountain_car_V(omega))
scatter(randomly_selected_trans[:,0],randomly_selected_trans[:,1])
traj=randomly_selected_trans

# <codecell>

psi=mountain_car_psi
phi=mountain_car_phi
s=traj[:,:2]
a=traj[:,2]
#Classification
from sklearn import svm
clf = svm.SVC(C=1000., probability=True)
clf.fit(s, a)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
#Plots de la politique de l'expert, des données fournies par l'expert, de la politique du classifieur
mountain_car_plot_policy(policy)
scatter(traj[:,0],traj[:,1],c=traj[:,2])
figure()
mountain_car_plot_policy(pi_c)
scatter(traj[:,0],traj[:,1],c=traj[:,2])
figure()
##Plots de Q et de la fonction de score du classifieur et évaluation de la politique du classifieur
#phi=inverted_pendulum_phi
Q = lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
Q_0 = lambda p,s:Q(zip_stack(p,s,-1*ones(p.shape)))
Q_1 = lambda p,s:Q(zip_stack(p,s,0*ones(p.shape)))
Q_2 = lambda p,s:Q(zip_stack(p,s,1*ones(p.shape)))
q_0 = lambda p,s:q(zip_stack(p,s,-1*ones(p.shape)))
q_1 = lambda p,s:q(zip_stack(p,s,0*ones(p.shape)))
q_2 = lambda p,s:q(zip_stack(p,s,1*ones(p.shape)))
mountain_car_plot(Q_0)
figure()
mountain_car_plot(Q_1)
figure()
mountain_car_plot(Q_2)
figure()
mountain_car_plot(q_0)
figure()
mountain_car_plot(q_1)
figure()
mountain_car_plot(q_2)

# <codecell>

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
#Plot des samples hat_r Pour chacune des 3 actions
sar = hstack([sa,hat_r])
for action in ACTION_SPACE:
    sr = array([l for l in sar if l[2]==action])
    axis([-1.2,0.6,-0.07,0.07])
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
    colorbar()
    figure()
##Avec l'heuristique : 
regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in ACTION_SPACE] 
def add_output_column( reg_mat ):
    actions = reg_mat[:,-1].reshape(column_shape)
    hat_r_bool_table = array(actions==a)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
    return hstack([reg_mat,output_column])
regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#On plotte les mêmes données que juste précedemment, mais avec l'heuristique en prime
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    axis([-1.2,0.6,-0.07,0.07])
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
    colorbar()
    figure()

# <codecell>

#Régression
from sklearn.svm import SVR
y = regression_matrix[:,-1]
X = regression_matrix[:,:-1]
reg = SVR(C=1.0, epsilon=0.2)
reg.fit(X, y)
CSI_reward = lambda sas:reg.predict(sas[:3])[0]
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
#On plotte les rewards en fonction de l'action
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    R = lambda p,s: squeeze( vCSI_reward(zip_stack(p,s,action*ones(p.shape),p,s)))
    pos = linspace(-1.2,0.6,30)
    speed = linspace(-0.07,0.07,30)
    pos,speed = meshgrid(pos,speed)
    Z = R(pos,speed)
    figure()
    contourf(pos,speed,Z,50)
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', )#cmap = cm.jet );
    clim(vmin=min(Z.reshape(-1)),vmax=max(Z.reshape(-1)))
    colorbar()
    figure()
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a,s,p) for a in actions]
    return mean(array([squeeze(vCSI_reward(m)) for m in matrices]), axis=0)
mountain_car_plot(mean_reward)

# <codecell>

#Evaluation de l'IRL
data = mountain_car_training_data(freward=CSI_reward)
policy_CSI,omega_CSI = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )

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

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))

plottable_episode_length = mountain_car_episode_vlength(policy)
X = linspace(-1.2,0.6,5)
Y = linspace(-0.07,0.07,5)
X,Y = meshgrid(X,Y)
Z = plottable_episode_length(X,Y)
contourf(X,Y,Z,50)
colorbar()
max(Z.reshape(-1))

