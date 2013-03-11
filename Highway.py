# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

def Sgenerator( ):
    for v in range(0,3):
        for x_b in range(0,9):
            for y_r in range(0,9):
                for x_r in range(0,3):
                    yield [v,x_b,y_r,x_r]

S = [s for s in Sgenerator()]

A = range(0,5)

def s_index( state ):
    v = state[0]
    x_b = state[1]
    y_r = state[2]
    x_r = state[3]
    index = x_r + y_r*3 + x_b*3*9 + v*3*9*9
    return index

def sa_index( state, action ):
    v = state[0]
    x_b = state[1]
    y_r = state[2]
    x_r = state[3]
    a = action
    index = x_r + y_r*3 + x_b*3*9 + v*3*9*9 + a*3*9*9*3
    return index

def next_states( state, action ):
    "Returns a tuple of the next possible states given the agent is in the provided state ant takes the provided action."
    v = next_v = state[0]
    xb = next_xb = state[1]
    yr = next_yr = state[2]
    xr = next_xr = state[3]
    #taking the player's action into account
    if action == 0:
        pass
    elif action == 1:
        next_v = v + 1 if v < 2 else 2
    elif action == 2:
        next_v = v - 1 if v > 0 else 0
    elif action == 3:
        next_xb = xb - 1 if xb > 0 else 0
    elif action == 4:
        next_xb = xb + 1 if xb < 8 else 8
    else:
        raise ValueError( "Action %d does not exist" % action )
    #Moving the red car
    next_yr_lst = []
    if v == 0:
        next_yr_lst = range(0,9)
    elif v == 1:
        next_yr_lst = [1,3,5,7]
    elif v == 2:
        next_yr_lst = [1,4,7]
    else:
        raise ValueError("Speed %d is unknown to me"%v)
    possible_outcomes = []        
    try:
        next_yr = (i for i in next_yr_lst if i > yr).next()
        possible_outcomes.append( [next_v, next_xb, next_yr, next_xr] )
    except StopIteration : #This means the car has reached past its final position
        next_yr = next_yr_lst[0]
        possible_outcomes = [ [next_v, next_xb, next_yr, i] for i in range(0,3) ]
    return possible_outcomes

def P( a ):
    "Returns the matrix of transition probability for action a."
    P_a = zeros((3*9*9*3,3*9*9*3))
    for state in Sgenerator():
        current_index = s_index( state )
        possible_outcomes = next_states( state, a )
        #Writing the probabilities in the matrix
        for next_s in possible_outcomes:
            next_index = s_index( next_s )
            P_a[ current_index, next_index ] = 1./len(possible_outcomes) #This line assumes two outcome won't share the same index
    return P_a

P = hstack([P(a) for a in A])
savetxt("Highway_P.mat",P)

# <codecell>

def R( ):
    reward = zeros((3*9*9*3,1))
    for state in S:
        current_index = s_index( state )
        v = state[0]
        xb = state[1]
        yr = state[2]
        xr = state[3]
        lane_nb2blue_x = [[1,2,3],[3,4,5],[5,6,7]] #Coincidentally, lane_nb is xr
        if yr in [6,7,8] and xb in lane_nb2blue_x[xr] : #Collision
            reward[ current_index ] = -1.
        elif xb in [0,1,7,8]:
            reward[ current_index ] = -0.5
        elif v == 2:
            reward[ current_index ] = 1.
        else:
            pass #already at 0
    return reward
savetxt("Highway_R.mat", R())

