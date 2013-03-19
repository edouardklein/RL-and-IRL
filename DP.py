from numpy import *
from stuff import *
import scipy
import pdb

#code from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
import random

def weighted_choice(choices):
   total = sum(w for c,w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto+w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"
#end of code from StackOverflow
class MDP:
    def __init__(self, P, R):
        print "MDP OK3"
        self.cardSA,self.cardS = P.shape
        self.cardA = self.cardSA/self.cardS
        self.gamma=0.9
        self.P = P
        assert all(f_eq(self.P.sum(axis=1), 1)), "Probability matrix unproperly conditionned"
        self.R = R
        #self.A = range(0,self.cardA)
        print "Shape of P "+str(P.shape)
        print "Shape of R "+str(R.shape)
        print "Card of S "+str(self.cardS)
        print "Card of A "+str(self.cardA)
        print "Card of SA "+str(self.cardSA)
        

    def optimal_policy(self):
        "Exact dynamic programming algorithm, the reward vector is over the state action space"
        V = ones((self.cardS,1))*(-Inf)
        Pi = zeros((self.cardS,self.cardSA))
        Pi[:,0:self.cardS] = identity(self.cardS) #Default policy, action 0 everywhere
        assert all(Pi.sum(axis=1) == 1), "A sum of probabilities should give 1."
        oldPi = Pi.copy()
        T=0
        while True: #Do..while
            oldPi = Pi.copy()
            oldV = V.copy()
            assert all(f_eq(dot(Pi,self.P).sum(axis=1),1)), "A sum of probabilities should give 1."
            V = linalg.solve( identity( self.cardS ) - self.gamma*dot(Pi,self.P), dot( Pi, self.R) )
            assert (V-oldV).min() >= -1e-10,"Greedy policy not better than old policy, min and max of diff are"+str([(V-oldV).min(),(V-oldV).max()])
            assert allclose(V, dot(Pi,self.R) + self.gamma*dot(dot(Pi,self.P),V)), "Bellman equation"
            Q = self.R + self.gamma*dot( self.P,V)
            Pi = self.Q2Pi( Q )
            print "Iteration "+str(T)+", "+str(sum(Pi!=oldPi))+"\tactions changed."
            T+=1
            if( all( Pi == oldPi ) ):
                break
        return Pi,V,lambda s: self.control(s,Pi)

    def Q2Pi(self, Q):
        #This assumes that sa_index(s,a) = s_index(s)+a*card(S)
        reshaped_Q = (Q.reshape((self.cardA,self.cardS))).transpose() #SAx1 -> SxA
        maxScore = dot(reshaped_Q.max(axis=1).reshape((self.cardS,1)),ones((1,self.cardA)))
        decision = f_eq(reshaped_Q,maxScore) #Multiple 'True' if ex-aequo
        answer = zeros((self.cardS,self.cardSA))
        for i in range(0,self.cardS):
            for j in range(0,self.cardA):
                sa_index = i+j*self.cardS
                if decision[i,j]:
                    answer[i,sa_index] = 1.
                    break #Breaking here arbitrarily choose the lowest indexed action to break ties
            assert all([ f_geq(Q[i+j*self.cardS], Q[i+a*self.cardS]) for a in range(0,self.cardA)]), r"$Q(s,pi(s)) = \arg\max_a Q(s,a)$"
        assert all(answer.sum(axis=1) == 1), "A sum of probabilities should give 1."
        return answer

    def control( self, s, pi ):
        choices = [(a,pi[s,s+a*self.cardS]) for a in range(0,self.cardA)]
        return weighted_choice( choices )
    
    def D_func(self, control, M, L ,rho, reward = None):
        "Returns M episodes of length L when acting according to control and starting according to rho"
        answer = []
        for m in range(0,M):
            s = rho()
            for l in range(0,L):
                a = control( s )
                s_dash = self.simul( s, a )
                r = 0
                if reward != None:
                    r = reward( s, a, s_dash )
                trans = [s,a,s_dash,r]
                answer.append( trans )
                s = s_dash
        return answer
    
    def simul( self, s, a ):
        sa_index = s + a*self.cardS
        choices = zip(range(0,self.cardS),self.P[sa_index])
        return weighted_choice( choices )

    def evaluate(self, Pi, R=None):
       "Returns $E[V^\pi_R(s)|s\in S]$"
       if R==None:
          R = self.R
       V = linalg.solve( identity( self.cardS ) - 0.9*dot(Pi,self.P), dot( Pi, R) )
       return mean(V)

