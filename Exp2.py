# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pendulum import *

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

# <codecell>

policy = inverted_pendulum_expert_policy()
#inverted_pendulum_episode_length(0,0,policy)
plottable_episode_length = inverted_pendulum_episode_vlength(policy)
plottable_episode_average_length = inverted_pendulum_episode_average_vlength(policy)

# <codecell>

X = linspace(-pi,pi,30)
Y = linspace(-pi,pi,30)
Y,X = meshgrid(X,Y)
Z = plottable_episode_length(X,Y)

# <codecell>

contourf(X,Y,Z,50)
colorbar()

# <codecell>

Z2 = plottable_episode_average_length(X,Y)

# <codecell>

contourf(X,Y,Z2,50)
colorbar()

# <codecell>

Z3=genfromtxt("Z.mat") #Distribution de l'expert
contourf(X,Y,Z2,50)
pos = linspace(-0.3,0.3,30)
speed = linspace(-2,2,30)
contour(pos,speed,Z3,50)

# <codecell>

omega_CSI = genfromtxt("omega_CSI.mat")
policy_CSI = greedy_policy(omega_CSI, inverted_pendulum_phi, ACTION_SPACE)
plottable_episode_length = inverted_pendulum_episode_vlength(policy_CSI)
plottable_episode_average_length = inverted_pendulum_episode_average_vlength(policy_CSI)
Z4 = plottable_episode_length(X,Y)
contourf(X,Y,Z4,50)
colorbar()

# <codecell>

clf = None
import pickle
with open('classification.obj', 'rb') as input:
    clf = pickle.load(input)
clf_predict= lambda state : clf.predict(squeeze(inverted_pendulum_psi(state)))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))

# <codecell>

plottable_episode_length = inverted_pendulum_episode_vlength(pi_c)
plottable_episode_average_length = inverted_pendulum_episode_average_vlength(pi_c)
Z5 = plottable_episode_length(X,Y)

# <codecell>

contourf(X,Y,Z5,50)
colorbar()

