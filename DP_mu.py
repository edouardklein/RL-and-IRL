
from numpy import *
import scipy

#FIXME: Cette fonction n'a rien a faire la, mais je la rangerai plus tard
def monte_carlo_mu( traj, gamma, psi ):
    #try:
    assert( len( traj.shape ) == 2 )
    #except AssertionError: #Trajectory with only one transition
    #    return psi( traj )
    mGamma = map( lambda i: pow( gamma,i),range(0,traj.shape[0]) )
    mPsi = map( psi, traj )
    mDiscountedPsi = map( lambda gamma, psi : gamma*psi, mGamma, mPsi )
    return sum( mDiscountedPsi, 0 )
    


g_fGamma = 0.9

def DP_mu( pi, Phi ):
    "Returns the matrix corresponding to the feature expectation of the given policy."
    global g_fGamma
    answer = scipy.rand( Phi.shape[0], Phi.shape[1] )
    changed = True
    while changed:
        new_answer = Phi + g_fGamma*dot(pi,answer)
        diff = sum( abs( new_answer - answer ) )
        answer = new_answer
        if diff > 0.001:
            changed = True
        else:
            changed = False
    return answer
