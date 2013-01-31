# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pendulum import *

# <codecell>

def inverted_pendulum_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < EXPERT_RUN_LENGTH and reward == 0. :
        action = policy(state)
        next_state = inverted_pendulum_next_state(state,action)
        reward = inverted_pendulum_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def inverted_pendulum_episode_vlength(policy):
    return vectorize(lambda p,s:inverted_pendulum_episode_length(p,s,policy))

# <codecell>

policy = inverted_pendulum_expert_policy()
#inverted_pendulum_episode_length(0,0,policy)
plottable_episode_length = inverted_pendulum_episode_vlength(policy)

# <codecell>

X = linspace(-pi,pi,30)
Y = linspace(-pi,pi,30)
Y,X = meshgrid(X,Y)
Z = plottable_episode_length(X,Y)
contourf(X,Y,Z,50)
colorbar()

# <codecell>

contourf(X,Y,Z,50)
colorbar()

# <codecell>

def inverted_pendulum_episode_average_length(initial_position,initial_speed,policy):
    return mean([inverted_pendulum_episode_length(initial_position,initial_speed, policy) for i in range(0,3)])

def inverted_pendulum_episode_average_vlength(policy):
    return vectorize(lambda p,s:inverted_pendulum_episode_average_length(p,s,policy))

# <codecell>


