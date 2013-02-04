# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/usr/bin/env python
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.integrate import dblquad

from pendulum import *

policy = inverted_pendulum_expert_policy()
Q = lambda sa: squeeze(dot(inverted_pendulum_expert_omega.transpose(), inverted_pendulum_phi(sa)))
def hatR(state):
    next_state = inverted_pendulum_next_state(state,policy(state))
    next_action = policy(next_state)
    sa = hstack([state,policy(state)])
    sa_dash = hstack([next_state,next_action])
    return Q(sa)-GAMMA*Q(sa_dash)
vhatR = non_scalar_vectorize(hatR,(2,),(1,1))
pos = linspace(-pi,pi,30)
speed = linspace(-pi,pi,30)
pos,speed=meshgrid(pos,speed)
S = zip_stack(pos,speed)
actions = policy(S)
#Z = Q(zip_stack(pos,speed,actions))
Z = squeeze(vhatR(zip_stack(pos,speed)))
Z.shape
figure()
contourf(pos,speed,Z,50)
colorbar()

# <codecell>

data = inverted_pendulum_trace(policy, run_length=100, initial_state=inverted_pendulum_expert_distribution_sample)

# <codecell>

#CSI, normal
s=data[:,:2]
a=data[:,2]
from sklearn import svm
clf = svm.SVC(C=1000., probability=True)
clf.fit(s, a)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
##DEBUG
with open('classification.obj', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
##END DEBUG
column_shape = (len(data),1)
s = data[:,0:2]
a = data[:,2].reshape(column_shape)
sa = data[:,0:3]
s_dash = data[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
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
data_for_lspi = inverted_pendulum_random_trace(reward=CSI_reward)
policy_CSI,omega_CSI = policy,omega = lspi( data_for_lspi, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=20 )
savetxt("omega_CSI.mat",omega_CSI)

# <codecell>

#CSI, sur S seul
s=data[:,:2]
a=data[:,2]
from sklearn import svm
clf = svm.SVC(C=1000., probability=True)
clf.fit(s, a)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
##DEBUG
with open('classification.obj', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
##END DEBUG
column_shape = (len(data),1)
s = data[:,0:2]
a = data[:,2].reshape(column_shape)
sa = data[:,0:3]
s_dash = data[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
regression_matrix = hstack([s,hat_r])
#Régression
from sklearn.svm import SVR
y = regression_matrix[:,-1]
X = regression_matrix[:,:-1]
reg = SVR(C=1.0, epsilon=0.2)
reg.fit(X, y)
CSI_reward = lambda sas:reg.predict(sas[:2])[0]
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
data_for_lspi = inverted_pendulum_random_trace(reward=CSI_reward)
policy_CSI,omega_CSI = policy,omega = lspi( data_for_lspi, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=20 )
savetxt("omega_CSI.mat",omega_CSI)

# <codecell>

#CSI, normal
s=data[:,:2]
a=data[:,2]
from sklearn import svm
clf = svm.SVC(C=1000., kernel='linear',probability=True)
clf.fit(inverted_pendulum_psi(s).reshape((100,10)), a)
clf_predict= lambda state : clf.predict(squeeze(inverted_pendulum_psi(state)))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(inverted_pendulum_psi(sa[:2]))))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
##DEBUG
with open('classification.obj', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
##END DEBUG
column_shape = (len(data),1)
s = data[:,0:2]
a = data[:,2].reshape(column_shape)
sa = data[:,0:3]
s_dash = data[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)#-0.*ones(column_shape)
regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in ACTION_SPACE] 
def add_output_column( reg_mat ):
    actions = reg_mat[:,-1].reshape(column_shape)
    hat_r_bool_table = array(actions==a)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
    return hstack([reg_mat,output_column])
regression_matrix = vstack(map(add_output_column,regression_input_matrices))
##SANS HEURISTIQUES
#regression_matrix = hstack([sa,hat_r])
#Régression
from sklearn.svm import SVR
y = regression_matrix[:,-1]
X = regression_matrix[:,:-1]
#reg = SVR(C=1000.0, epsilon=.5, kernel='rbf')
#reg = SVR()
reg = SVR(kernel='linear')
reg.fit(inverted_pendulum_phi(X).reshape((300,30)), y)
#CSI_reward = lambda sas:reg.predict(squeeze(inverted_pendulum_phi(sas[:3])))[0]
CSI_reward = lambda sas:q(sas[:3])
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
data_for_lspi = inverted_pendulum_random_trace(reward=CSI_reward)
policy_CSI,omega_CSI = policy,omega = lspi( data_for_lspi, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=20 )
savetxt("omega_CSI.mat",omega_CSI)
#On plotte les rewards en fonction de l'action
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    R = lambda p,s: squeeze( vCSI_reward(zip_stack(p,s,action*ones(p.shape),p,s)))
    pos = linspace(-pi,pi,30)
    speed = linspace(-pi,pi,30)
    pos,speed = meshgrid(pos,speed)
    Z = R(pos,speed)
    figure()
    contourf(pos,speed,Z,50)
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', )#cmap = cm.jet );
    clim(vmin=min(Z.reshape(-1)),vmax=max(Z.reshape(-1)))
    colorbar()
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a,s,p) for a in actions]
    return mean(array([squeeze(vCSI_reward(m)) for m in matrices]), axis=0)
inverted_pendulum_plot(mean_reward)

# <codecell>

#Training GMM to build features
#data_0data[data[:,2]==0.,:3]
from sklearn import mixture
gmm0 = mixture.GMM(n_components=1, covariance_type='full')
gmm1 = mixture.GMM(n_components=1, covariance_type='full')
gmm2 = mixture.GMM(n_components=1, covariance_type='full')
gmm0.fit(data[data[:,2]==0.,:2])
gmm1.fit(data[data[:,2]==1.,:2])
gmm2.fit(data[data[:,2]==2.,:2])
def single_psi(sa):
    return array([exp(gmm.score(sa[:,:2])) for gmm in [gmm0,gmm1,gmm2]])
def single_phi(sa):
    action = int(sa[-1])
    sa = sa.reshape(1,sa.shape[0])
    answer = zeros((9,1))
    answer[action*3:action*3+3] = single_psi(sa)
    return answer

phi = non_scalar_vectorize(single_phi,(3,),(9,1))
    
X = linspace(-1,1,60)
Y = linspace(-1,1,60)
Y,X = meshgrid(X,Y)
Act = 2*ones(X.shape)
SA = zip_stack(X,Y,Act)
print SA.shape
phi_sa = phi(SA.reshape((60*60,3))).reshape((60,60,9))
#phi_sa.transpose().shape
Z = dot(phi_sa,array([1,1,1,1,1,1,1,1,1]))
contourf(X,Y,Z,50)
colorbar()

# <codecell>

#CSI, normal
s=data[:,:2]
a=data[:,2]
from sklearn import svm
clf = svm.SVC(C=1000., probability=True)
clf.fit(s, a)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q= Q
#q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
##DEBUG
with open('classification.obj', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
##END DEBUG
column_shape = (len(data),1)
s = data[:,0:2]
a = data[:,2].reshape(column_shape)
sa = data[:,0:3]
s_dash = data[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in ACTION_SPACE] 
def add_output_column( reg_mat ):
    actions = reg_mat[:,-1].reshape(column_shape)
    hat_r_bool_table = array(actions==a)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
    return hstack([reg_mat,output_column])
regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#Régression
max(hat_r)

# <codecell>

#from sklearn.svm import SVR
from sklearn import linear_model
y = regression_matrix[:,-1]
X = squeeze(phi(regression_matrix[:,:-1]))
print X.shape
#reg = SVR(C=1000.0, epsilon=0.1,kernel='linear')
#reg = linear_model.LinearRegression()
#reg.fit(X, y)
reg_omega = dot(dot(inv(dot(X.transpose(),X)),X.transpose()),y).reshape((9,1))
#reg_omega = array([1,1,1,1,1,1,1,1,1])
#print reg_omega.shape


#CSI_reward = lambda sas:dot(squeeze(phi(sas[:3])),reg_omega)[0]
#CSI_reward = lambda sas:Q(sas[:3])#Marchotte
CSI_reward = lambda sas:Q(sas[:3])-GAMMA*Q(hstack([sas[-2:],policy(sas[-2:])]))#Moins bien
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
data_for_lspi = inverted_pendulum_random_trace(reward=CSI_reward)
policy_CSI,omega_CSI = lspi( data_for_lspi, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=20 )
savetxt("omega_CSI.mat",omega_CSI)
#On plotte les rewards en fonction de l'action
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    R = lambda p,s: squeeze( vCSI_reward(zip_stack(p,s,action*ones(p.shape),p,s)))
    pos = linspace(-0.2,0.2,30)
    speed = linspace(-1,1,30)
    pos,speed = meshgrid(pos,speed)
    Z = R(pos,speed)
    figure()
    
    contourf(pos,speed,Z,50)
    clim(vmin=min(sr[:,3].reshape(-1)),vmax=max(sr[:,3].reshape(-1)))
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', )#cmap = cm.jet );
    colorbar()
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a,s,p) for a in actions]
    return mean(array([squeeze(vCSI_reward(m)) for m in matrices]), axis=0)
inverted_pendulum_plot(mean_reward)

# <codecell>


X = linspace(-pi,pi,30)
Y = linspace(-pi,pi,30)
Y,X = meshgrid(X,Y)
#policy_CSI=lambda s:choice(ACTION_SPACE)
plottable_episode_length = inverted_pendulum_episode_vlength(policy_CSI)
plottable_episode_average_length = inverted_pendulum_episode_average_vlength(policy_CSI)
Z4 = plottable_episode_length(X,Y)
contourf(X,Y,Z4,50)
colorbar()

# <codecell>

def inverted_pendulum_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    #while answer < EXPERT_RUN_LENGTH and reward == 0. :
    while answer < 100 and reward == 0. :
        action = policy(state)
        next_state = inverted_pendulum_next_state(state,action)
        reward = inverted_pendulum_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def inverted_pendulum_episode_vlength(policy):
    return vectorize(lambda p,s:inverted_pendulum_episode_length(p,s,policy))

def inverted_pendulum_episode_average_length(initial_position,initial_speed,policy):
    return mean([inverted_pendulum_episode_length(initial_position,initial_speed, policy) for i in range(0,10)])

def inverted_pendulum_episode_average_vlength(policy):
    return vectorize(lambda p,s:inverted_pendulum_episode_average_length(p,s,policy))

