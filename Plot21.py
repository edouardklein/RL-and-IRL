# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2
from DP import *
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

def next_states( state, action ):
    "Returns the list [[s,w],...] of next possible states and associated probability"
    x = state[0]
    y = state[1]
    x_south = x
    y_south = y + 1 if y!=4 else 4
    x_west = x - 1 if x!=0 else 0
    y_west = y
    x_east = x + 1 if x!=4 else 4
    y_east = y 
    x_north = x
    y_north = y - 1 if y!=0 else 0
    weights = zeros((1,4)) + .1
    weights[0,action] = 0.7
    assert abs(sum(weights) - 1.) < 0.00001
    states = map( array, [[x_south,y_south],[x_west,y_west],[x_east,y_east],[x_north,y_north]]) #Same order as specified in the textual description of the action space
    return zip( states, weights[0] )
def P( a ):
    "Returns the matrix of transition probability for action a."
    P_a = zeros((5*5,5*5))
    for state in Sgenerator():
        current_index = s_index( state )
        states = next_states( state, a )
        for sdash,w in states:
            index_dash = s_index( sdash )
            P_a[current_index, index_dash] += w
    return P_a
def Sgenerator( ):
    for x in range(0,5):
        for y in range(0,5):
            yield [x,y]

def s_index( state ):
    x = state[0]
    y = state[1]
    index = y*5 + x
    return int(index)
def sa_index( state, action ):
    x = state[0]
    y = state[1]
    a = action
    index = y*5*4 + x*4 + a
    return int(index)
P = vstack([P(a) for a in range(0,4)])
def R():
    reward = zeros((5*5*4,1))
    indices = [sa_index([4,0],a) for a in range(0,4)]
    reward[indices] = 1.
    return reward
GW = MDP(P,R())
GW.optimal_policy()

