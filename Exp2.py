from numpy import *
import scipy
from TT_DP import *
from a2str import *
from TT import *
import sys

m_E = 4
m_A = 4
for n in range(25,26):
    meanNbRewards = 0
    for j in range(0,10):
        R = scipy.rand( n )
        ExpertsActions = []
        for i in range(0,m_E):
            P_i = scipy.rand(n,n)
            for line in P_i:
                line /= sum(line) #Sum of proba = 1, so we normalize the random line
            ExpertsActions.append(P_i)
        P_pi = TT_DP( R, ExpertsActions )
        ttRewards = TT( P_pi, ExpertsActions )
        nbRewards = 0
        if( ttRewards == None ):
            nbRewards = 0
        elif( len( ttRewards.shape) == 1 ): #If there is only one reward
            ttRewards = asarray([ttRewards]) #Cast as a matrix anyway, the code below expects a matrix and not a vector
            nbRewards = 1
        else:
            nbRewards = ttRewards.shape[0]
        sys.stderr.write("n = %d, j=%d, nbRewards = %d\n"%(n,j,nbRewards))
        meanNbRewards+=nbRewards
    meanNbRewards/=10.
    print "%d %f" % (n,meanNbRewards)
