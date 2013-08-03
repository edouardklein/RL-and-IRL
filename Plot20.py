# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


from numpy import *
import scipy
import sys
from DP import *
from DP_mu import *
from RWC import *
import random

class GridWorld(MDP):
    card_S = 25
    card_A = 4
    
    def Sgenerator(self):
        for x in range(0,5):
            for y in range(0,5):
                yield [x,y]
    S = None
    A = range(0,4)
    
    def s_index(self, state ):
        x = state[0]
        y = state[1]
        index = y*5 + x
        return int(index)

    def sa_index(self, state, action ):
        x = state[0]
        y = state[1]
        a = action
        index = y*5*4 + x*4 + a
        return int(index)
    
    def P(self, a):
        "Returns the matrix of transition probability for action a."
        P_a = zeros((5*5,5*5))
        for state in self.Sgenerator():
            current_index = self.s_index( state )
            states = next_states( state, a )
            for sdash,w in states:
                index_dash = self.s_index( sdash )
                P_a[current_index, index_dash] += w
        return P_a

    def R(self):
        reward = zeros((5*5*4,1))
        indices = [self.sa_index([4,0],a) for a in self.A]
        reward[indices] = 1.
        return reward

    
    def __init__(self):
        self.S = [s for s in self.Sgenerator()]
        m_P = vstack([self.P(a) for a in self.A])
        R = self.R()
        super().__init__(m_P,R)



def psi( s ):
    answer = zeros(( 5*5, 1 ))
    answer[ s_index( s )] = 1.
    return answer

def phi( s,a ):
    answer = zeros(( 5*5*4, 1 ))
    answer[ sa_index( s, a )] = 1.
    return answer 


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



def GWDP( R, filename ):
    return DP( R, S, s_index, A, [P(a) for a in A], lambda x:x, sa_index, filename )

def GWDPSA( R, filename ):
    return DPSA( R, S, s_index, A, [P(a) for a in A], lambda x:x, sa_index, filename )

def evaluate_Pi( Pi ):
    sys.stderr.write( "Mu computation...\n" )
    Mu = DP_mu( Pi, identity( 5*5 ))
    mean_Mu = mean( Mu, 0 )
    return dot( mean_Mu, R() )

def evaluate_theta( theta, l_psi ):
    dicR = {}
    for s in Sgenerator():
        index = s_index( s )
        dicR[ index ] = dot( theta.transpose(), l_psi( s ) )
    R_theta = zeros(( len(dicR), 1 ))
    for i in dicR:
        R_theta[ i ] = dicR[ i ]
    sys.stderr.write( "Pi computation...\n" )
    Pi = GWDP( R_theta, "V_agent.mat" )
    return evaluate_Pi( Pi )

def evaluate_thetaSA( theta, l_phi ):
    dicR = {}
    for s in Sgenerator():
        for a in A:
            index = sa_index( s,a )
            dicR[ index ] = dot( theta.transpose(), l_phi( s, a ) )
    R_theta = zeros(( len(dicR), 1 ))
    for i in dicR:
        R_theta[ i ] = dicR[ i ]
    sys.stderr.write( "Pi computation...\n" )
    Pi = GWDPSA( R_theta, "V_agent.mat" )
    return evaluate_Pi( Pi )

def omega_play( omega, L, M ):
    "Plays M episodes of length L, actig according to the greedy policy described by omega. Returns the transitions."
    answer = zeros(( L*M, 2+1+2+1+1 ))
    reward  = R()
    for iep in range(0,M):
        #state = array(map( int, array([5,5])*scipy.rand(2)))
        state = array([0,4])
        eoe = 1
        itrans = 0
        while eoe == 1:
            action = greedy_policy( state, omega, phi, A )
            next_state = weighted_choice( next_states( state, action ))
            r = reward[ s_index( state ) ]
            eoe = 0 if itrans >= L-1 or (state[0]==4 and state[1]==0) else 1 #0 means end of episode.
            index = iep*L + itrans
            trans = []
            [ trans.extend(i) for i in [state, [action], next_state, [r, eoe] ]]
            answer[ index, : ] = trans
            state = next_state
            itrans+=1
    return answer

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

print("Expert creation...")
GW = GridWorld()
Pi_E,V_E,f_Pi_E = GW.optimal_policy()
Pi_E,V_E,f_Pi_E 

# <codecell>

V_E.reshape((5,5))
array(range(0,25)).reshape((5,5))

# <codecell>

#Histogramme de la valeur
my_cmap = matplotlib.pyplot.cm.cubehelix
B = array(range(0,25))
A = zeros((5,5))
for x in range(0,5):
    for y in range(0,5):
        A[4-y,x] = V_E[GW.s_index([x,y])]
imshow(A, interpolation='none',cmap=my_cmap)
colorbar()
A

# <codecell>

#x,y = meshgrid(range(0,5),range(0,5))
#scatter(x,y,c=[f_Pi_E(GW.s_index(s)) for s in zip_stack(x,y).reshape(-1,2)],)
gca().invert_yaxis()
#zip_stack(x,y).reshape(-1,2)
#B = [f_Pi_E(GW.s_index(s)) for s in ]
A = zeros((5,5))
for x in range(0,5):
    for y in range(0,5):
        A[4-y,x] = f_Pi_E(GW.s_index([x,y]))
imshow(A, interpolation='none',cmap=my_cmap)
colorbar()

