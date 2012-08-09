import sys
from LAFEM import *

#Small util, needed because arrays are not hashable and thus cannot be directly used in a python dictionary
def l2str(l):
	"""Return the unique string representing line l"""
	answer = ""
	for x in l:
		if (abs(x)<1e-10): #FIXME : this is not right.
			answer += " 0.00e+00\t"
		elif (x>0):
			answer += " %1.2e\t"%x
		else:
			answer += "%+1.2e\t"%x
	answer +="\n"
	return answer

#Simple and straightforward way to evaluate mu_E
def monte_carlo_mu( traj, gamma, psi ):
    #try:
    assert( len( traj.shape ) == 2 )
    #except AssertionError: #Trajectory with only one transition
    #    return psi( traj )
    mGamma = map( lambda i: pow( gamma,i),range(0,traj.shape[0]) )
    mPsi = map( psi, traj )
    mDiscountedPsi = map( lambda gamma, psi : gamma*psi, mGamma, mPsi )
    return sum( mDiscountedPsi, 0 )

#Tabular feature over the state space
def s_index( state ):
    x = state[0]
    y = state[1]
    index = y*5 + x
    return int(index)

def psi( s ):
    answer = zeros(( 5*5, 1 ))
    answer[ s_index( s )] = 1.
    return answer


#The Code of the SCIRL algorithm takes the form of a python class. Some of its members are left for the user to define. Thus, the user should subclass the LAFEM class and add the required members.
class SCIRL( LAFEM ):
    dicPi_E = {}
    A = range(0,4) #4 actions, this should match the convention the expert's transitions have been written with
    dicMu_E_s_only = {}

    def __init__( self ):
        #the member named data should be a list of tuples [s,a]. We extract these from the transition matrix.
        D_E = genfromtxt(sys.argv[1]) 
        for trans in D_E:
            self.dicPi_E[l2str(trans[0:2])] = trans[2:3][0]
            self.data = self.data +[[ trans[0:2], trans[2:3][0]]]
        
        #The results of this computation are going to be used in the mu_E function. See the tutoriel for more details
        dicMu_E_data = {}
        for start_index in range(0,len(D_E)):
            s = D_E[ start_index, 0:2 ]
            end_index = (i for i in range(start_index,len(D_E)) if D_E[i,2+1+2+1] == 0).next() #till next eoe
            data_MC = D_E[start_index:end_index+1,0:2]
            try:
                dicMu_E_data[l2str(s)].append(data_MC)
            except KeyError:
                dicMu_E_data[l2str(s)] = [data_MC]
        #Now dicMu_E_data contains the data that allows for the Monte-Carlo computation
        self.dicMu_E_s_only = {}
        for state in dicMu_E_data:
            gamma = 0.9
            lstMu_s = map( lambda traj: monte_carlo_mu( traj, 0.9, psi ), dicMu_E_data[state] )
            mu_s = mean( lstMu_s, 0 )
            self.dicMu_E_s_only[state] = mu_s

    #One should provide a mu_E function that computes the feature expectation of the expert for a given s and a
    def mu_E( self, s, a ):
        gamma = 0.9
        if self.dicPi_E[l2str(s)] == a:
            return self.dicMu_E_s_only[l2str(s)]
        else:
            return gamma*self.dicMu_E_s_only[l2str(s)]

    #This function is used by our gradient descent, see the paper for more information. The function given here is a sensible default that have empirically been shown to work wherever we tested it.
    def l( self, s, a ):
        if self.dicPi_E[l2str(s)] == a:
            return 0
        else:
            return 1
    #Here are the parameters for the gradient descent, they could need tweaking to work on the user's particular problem
    def alpha( self, t ):
        return 10./(t+1.)
    theta_0 = zeros( (25, 1) )
    Threshold = 0.0001
    T = 40

#Instantiation of the class we just defined, 
scirl = SCIRL()
#The algorithm is run
theta_scirl = scirl.run()
#And the result is printed
savetxt( "/dev/stdout", theta_scirl )
