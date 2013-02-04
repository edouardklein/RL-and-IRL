# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Misc.

# <codecell>

#!/usr/bin/env python
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.integrate import dblquad

from pendulum import *

# <headingcell level=2>

# Experiment 1 Code

# <codecell>

psi=inverted_pendulum_psi
phi=inverted_pendulum_phi
true_reward=lambda s,p:inverted_pendulum_reward(zip_stack(zeros(s.shape),zeros(s.shape),zeros(s.shape),s,p))
inverted_pendulum_plot(true_reward)
#Defining the expert policy
data_random = inverted_pendulum_random_trace()
data_expert,policy,omega = inverted_pendulum_expert_trace(inverted_pendulum_reward)

# <codecell>

#Defining the expert's stationary distribution
#On peut jouer avec la longueur d'un run et le nombre de runs
trajs = vstack([inverted_pendulum_trace(policy, run_length=60) for i in range(0,20)])
plot(trajs[:,0],trajs[:,1],ls='',marker='o')
axis([-10,10,-10,10])

# <codecell>

#On génère des runs de longueur suffisante, on vire les quelques premiers échantillons, et on prend un échantillon sur quelques uns par la suite
trajs = [inverted_pendulum_trace(policy, run_length=1000) for i in range(0,2)]
sampled_trajs = [t[100:999:10,:] for t in trajs]
expert_distrib_samples = vstack([t[:,-3:-1] for t in sampled_trajs])
plot(expert_distrib_samples[:,0],expert_distrib_samples[:,1],ls='',marker='o')
#axis([-10,10,-10,10])

# <codecell>

from sklearn import mixture
rho_E = mixture.GMM(covariance_type='full')
rho_E.fit(expert_distrib_samples)

pos = linspace(-0.3,0.3,30)
speed = linspace(-2,2,30)
pos,speed = meshgrid(pos,speed)
#g.score(zip_stack(pos,speed).reshape((30*30,2))).shape
Z = exp(rho_E.score(zip_stack(pos,speed).reshape((30*30,2)))).reshape((30,30))
#Z = two_arg_gaussian(pos,speed)
fig = figure()
contourf(pos,speed,Z,50)
colorbar()
scatter(expert_distrib_samples[:,0],expert_distrib_samples[:,1],s=1)
rho_E.sample()
import pickle
with open('inverted_pendulum_expert_distribution.obj', 'wb') as output:
    pickle.dump(rho_E, output, pickle.HIGHEST_PROTOCOL)
rho_E.get_params(deep=True)
rho_E

# <codecell>

#Données : 
traj = inverted_pendulum_trace(policy, run_length=300)
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
inverted_pendulum_plot_policy(policy)
scatter(traj[:,0],traj[:,1],c=traj[:,2])
inverted_pendulum_plot_policy(pi_c)
scatter(traj[:,0],traj[:,1],c=traj[:,2])
##Plots de Q et de la fonction de score du classifieur et évaluation de la politique du classifieur
#phi=inverted_pendulum_phi
Q = lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
Q_0 = lambda p,s:Q(zip_stack(p,s,0*ones(p.shape)))
Q_1 = lambda p,s:Q(zip_stack(p,s,1*ones(p.shape)))
Q_2 = lambda p,s:Q(zip_stack(p,s,2*ones(p.shape)))
q_0 = lambda p,s:q(zip_stack(p,s,0*ones(p.shape)))
q_1 = lambda p,s:q(zip_stack(p,s,1*ones(p.shape)))
q_2 = lambda p,s:q(zip_stack(p,s,2*ones(p.shape)))
inverted_pendulum_plot(Q_0)
inverted_pendulum_plot(Q_1)
inverted_pendulum_plot(Q_2)
inverted_pendulum_plot(q_0)
inverted_pendulum_plot(q_1)
inverted_pendulum_plot(q_2)
with open('classification.obj', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)

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
    axis([-pi,pi,-pi,pi])
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
    axis([-pi,pi,-pi,pi])
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

#Evaluation de l'IRL
data_CSI,policy_CSI,omega_CSI = inverted_pendulum_expert_trace(CSI_reward)

# <codecell>

savetxt("omega_CSI.mat",omega_CSI)

# <codecell>

#Critères de performance pour l'imitation
GAMMAS = array([pow(GAMMA,n) for n in range(0,70)])
def imitation_performance(policy):
    trajs = [inverted_pendulum_trace(policy, run_length=70) for i in range(0,100)]
    values = [sum(traj[:,5]*GAMMAS) for traj in  trajs]
    return mean(values)

#imitation_performance_policy(policy)

# <codecell>

#Critère de performance pour l'IRL
trajs_IRL = [inverted_pendulum_trace(policy_CSI, run_length=70, initial_state=lambda:rho_E.sample().reshape((2,))) for i in range(0,100)]
rewards_IRL = [vCSI_reward(t[:,:5]) for t in trajs_IRL]
value_IRL = mean([sum(r*GAMMAS) for r in rewards_IRL]) #V^*_{\hat R_C}
def IRL_performance(policy):
    trajs = [inverted_pendulum_trace(policy, run_length=70) for i in range(0,100)]
    rewards = [vCSI_reward(t[:,:5]) for t in trajs]
    value = mean([sum(r*GAMMAS) for r in rewards])
    return value_IRL-value

# <codecell>

print "Critere uniforme, expert :\t"+str(imitation_performance(policy))
print "Critere uniforme, classifieur :\t"+str(imitation_performance(pi_c))
print "Critere uniforme, IRL :\t"+str(imitation_performance(policy_CSI))

# <codecell>

print "Critere de la borne, expert :\t"+str(IRL_performance(policy))
print "Critere de la borne, classifieur :\t"+str(IRL_performance(pi_c))
print "Critere de la borne, IRL :\t"+str(IRL_performance(policy_CSI))

# <codecell>

def MC_R_C(state, action):
    next_states = [inverted_pendulum_next_state(state, action)]
    for i in range(0,10):
        next_states.append(inverted_pendulum_next_state(state, action))
    return q(hstack([state, action])) -GAMMA*mean([q(hstack([s,pi_c(s)])) for s in next_states])
def epsilon_R_pi(state, action):
    return MC_R_C(state, action) - reg.predict(hstack([state,action]))
def epsilon_R():
    return mean([max([epsilon_R_pi(state, a) for a in ACTION_SPACE]) for state in samples[:1000,:]])

# <codecell>

print "Abcisse possible, nb samples :\t"+str(traj.shape[0])
samples=rho_E.sample(7000)
sampled_pi_E = policy(samples)
sampled_pi_C = pi_c(samples)
print "Abcisse possible, epsilon_C :\t"+str(sum(sampled_pi_C != sampled_pi_E)/7000.)
#Epsilon R est techniquement calculable, mais pas franchement simple.
print "Abcisse possible, epsilon_R :\t"+str(epsilon_R())

# <codecell>

#Jouons avec la précision de l'estimation des grandeurs intéressantes
#Critère uniforme :
#GAMMAS = array([pow(GAMMA,n) for n in range(0,70)])
TRAJ_BAG_EXPERT = [inverted_pendulum_trace(policy, run_length=70) for i in range(0,2000)]
#TRAJ_BAG_EXPERT.extend([inverted_pendulum_trace(policy, run_length=70) for i in range(0,8000)])
#len(TRAJ_BAG_EXPERT)
#trajs = [inverted_pendulum_trace(policy, run_length=70) for i in range(0,2000)]
#value = mean([sum(t[:,5]*GAMMAS) for t in  trajs])
for i in range(0,1):
    X=map(int,linspace(10,2000,100))
    Y=[mean([sum(t[:,5]*GAMMAS) for t in  _t]) for _t in [TRAJ_BAG_EXPERT[i*2000:x+i*2000] for x in X]]
    plot(X,Y)
#On va supposer que 2000 trajectoires de 70 de long donnent un résultat à ~5% près

# <codecell>

#Critère de l'IRL
TRAJ_BAG_EXPERT_RHO=[inverted_pendulum_trace(policy, run_length=70, initial_state=lambda:rho_E.sample().reshape((2,))) for i in range(0,1000)]
print len(TRAJ_BAG_EXPERT_RHO)
#TRAJ_BAG_EXPERT_RHO.extend([inverted_pendulum_trace(policy, run_length=70, initial_state=lambda:rho_E.sample().reshape((2,))) for i in range(0,1000)])
print len(TRAJ_BAG_EXPERT_RHO)
rewards = [vCSI_reward(t[:,:5]) for t in TRAJ_BAG_EXPERT_RHO]
#rewards.extend([vCSI_reward(t[:,:5]) for t in TRAJ_BAG_EXPERT_RHO[-1000:]])
print len(rewards)
for i in range(0,1):
    X=map(int,linspace(10,1000,100))
    Y=[mean([sum(r*GAMMAS) for r in  _r]) for _r in [rewards[i*1000:x+i*1000] for x in X]]
    plot(X,Y)

#trajs_IRL = [inverted_pendulum_trace(policy_CSI, run_length=70, initial_state=lambda:rho_E.sample().reshape((2,))) for i in range(0,100)]
#rewards_IRL = [vCSI_reward(t[:,:5]) for t in trajs_IRL]
#value_IRL = mean([sum(r*GAMMAS) for r in rewards_IRL]) #V^*_{\hat R_C}
#def IRL_performance(policy):
#    trajs = [inverted_pendulum_trace(policy, run_length=70) for i in range(0,100)]
#    rewards = [vCSI_reward(t[:,:5]) for t in trajs]
#    value = mean([sum(r*GAMMAS) for r in rewards])
#    return value_IRL-value

