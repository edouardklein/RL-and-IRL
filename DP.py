from numpy import *
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
        self.R = R
        #self.A = range(0,self.cardA)
        print "Shape of P "+str(P.shape)
        print "Shape of R "+str(R.shape)
        print "Card of S "+str(self.cardS)
        print "Card of A "+str(self.cardA)
        print "Card of SA "+str(self.cardSA)
        

    def optimal_policy(self):
        "Exact dynamic programming algorithm, the reward vector is over the state action space"
        V = zeros((self.cardS,1))
        Pi = zeros((self.cardS,self.cardSA))
        oldPi = Pi.copy()
        T=0
        while True: #Do..while
            oldPi = Pi.copy()
            V = linalg.solve( identity( self.cardS ) - self.gamma*dot(Pi,self.P), dot( Pi, self.R) )
            Q = self.R + self.gamma*dot( self.P,V)
            Pi = self.Q2Pi( Q )
            print "Iteration "+str(T)+", "+str(sum(Pi!=oldPi))+"\tactions changed."
            if( all( Pi == oldPi ) ):
                break
        return Pi,V,lambda s: control(s,Pi)

    def Q2Pi(self, Q):
        #This assumes that sa_index(s,a) = s_index(s)+a*card(S)
        reshaped_Q = (Q.reshape((self.cardA,self.cardS))).transpose() #SAx1 -> SxA
        maxScore = dot(reshaped_Q.max(axis=1).reshape((self.cardS,1)),ones((1,self.cardA)))
        decision = reshaped_Q==maxScore #Multiple choices if ex-aequo
        answer = zeros((self.cardS,self.cardSA))
        for i in range(0,self.cardS):
            for j in range(0,self.cardA):
                sa_index = i+j*self.cardS
                if decision[i,j]:
                    answer[i,sa_index] = 1.
                    break #Breaking here arbitrarily choose the lowest indexed action to break ties
        return answer

    def control( self, s, pi ):
        choices = [(a,pi[s,s+a*self.cardS]) for a in range(0,self.cardA)]
        return weighted_choice( choices )
