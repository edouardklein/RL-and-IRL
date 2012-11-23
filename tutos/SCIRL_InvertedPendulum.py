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

def psi( s ):
    answer = zeros((10,1))
    position = s[0]
    speed = s[1]
    index = 0
    answer[index] = 1.
    index+=1
    for i in range(-1,2):
        for j in range(-1,2):
            d_i = i*3.141592/4.
            answer[index] = exp(-(pow(position-d_i,2) +
                                  pow(speed-j,2))/2.)
            index+=1
    return answer

def phi( s, a ):
    answer = zeros((30,1))
    index = a*10
    answer[ index:index+10 ] = psi( s )
    return answer


class SCIRL( LAFEM ):
    omega_mu_E = []
    dicPi_E = {}

    A = [0,1,2]

    def __init__( self ):
        D_E = genfromtxt(sys.argv[1])
        for trans in D_E:
            self.dicPi_E[l2str(trans[0:2])] = trans[2:3][0]
            self.data = self.data +[[ trans[0:2], trans[2:3][0]]]
        self.omega_mu_E = genfromtxt( sys.argv[2] )

    def l( self, s, a ):
        if self.dicPi_E[l2str(s)] == a:
            return 0
        else:
            return 1
    def mu_E( self, s, a ):
        answer = dot( self.omega_mu_E.transpose(), phi( s, a ) )
        return answer

    def alpha( self, t ):
        return 10./(t+1.)

    theta_0 = zeros( (10, 1) ) 

    Threshold = 0.2
    T = 20

scirl = SCIRL()
theta = scirl.run()

savetxt( "/dev/stdout", theta, "%e", "\n" );
