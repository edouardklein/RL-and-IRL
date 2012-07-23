from pylab import *

STATE_DIM = 6
ACTION_CARD = 4
PSI_DIM = 972
PHI_DIM = PSI_DIM*ACTION_CARD
g_aCards = array([3,4,3,3,3,3])
    
def psi( s ): #Tabular representation
    answer = zeros( [PSI_DIM, 1] )
    index = 0
    for i in range(0,STATE_DIM):
        index += s[i] * prod( g_aCards[0:i] )
    answer[index] = 1.
    return answer

def phi( s, a ):
    answer = zeros([PHI_DIM, 1])
    index = int(a)*PSI_DIM
    answer[index:index+PSI_DIM] = psi( s )
    return answer
