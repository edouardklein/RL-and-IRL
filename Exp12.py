# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

#SCIRL on the Highway
from DP import *
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *
P = genfromtxt("Highway_P.mat")
R = genfromtxt("Highway_R.mat")
Gamma = 0.9
ACTION_SPACE = range(0,5)

# <codecell>

Highway = MDP(P,R)
mPi_E, V_E, Pi_E = Highway.optimal_policy()

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
D_E = array(Highway.D_func(Pi_E, 1, 100,  rho))

# <codecell>

#SCIRL
#Precomputing mu with MC and heuristics
def MC_mu(episode):
    answer = zeros((729,1))
    for i in range(0,len(episode)):
        answer[episode[i]] += pow(Gamma,i)
    return answer

feature_expectations_MC = {}
for start_index in range(0,len(D_E)):
    data_MC=D_E[start_index:,:2]
    GAMMAS = range(0,len(data_MC))
    GAMMAS = array(map( lambda x: pow(Gamma,x), GAMMAS))
    state_action = data_MC[0,:2]
    state = data_MC[0,0]
    action = data_MC[0,1]
    mu = MC_mu(data_MC[:,0])
    feature_expectations_MC[str(state_action)] = mu
    for other_action in [a for a in ACTION_SPACE if a != action]:
        state_action=hstack([state,other_action])
        feature_expectations_MC[str(state_action)]=Gamma*mu
        

# <codecell>

#Structured Classifier
class GradientDescent(object):
    
   def alpha( self, t ):
      raise NotImplementedError, "Cannot call abstract method"

   theta_0=None
   Threshold=None
   T = -1
   sign = None
        
   def run( self, f_grad, f_proj=None, b_norm=False ): #grad is a function of theta
      theta = self.theta_0.copy()
      best_theta = theta.copy()
      best_norm = float("inf")
      best_iter = 0
      t=0
      while True:#Do...while loop
         t+=1
         DeltaTheta = f_grad( theta )
         current_norm = norm( DeltaTheta )
         if b_norm and  current_norm > 0.:
             DeltaTheta /= norm( DeltaTheta )
         theta = theta + self.sign * self.alpha( t )*DeltaTheta
         if f_proj:
             theta = f_proj( theta )
         print "Norme du gradient : "+str(current_norm)+", pas : "+str(self.alpha(t))+", iteration : "+str(t)

         if current_norm < best_norm:
             best_norm = current_norm
             best_theta = theta.copy()
             best_iter = t
         if current_norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print "Gradient de norme : "+str(best_norm)+", a l'iteration : "+str(best_iter)
      return best_theta


class StructuredClassifier(GradientDescent):
    sign=-1.
    Threshold=0.1 #Sensible default
    T=40 #Sensible default
    phi=None
    phi_xy=None
    inputs=None
    labels=None
    label_set=None
    dic_data={}
    x_dim=None
    
    def alpha(self, t):
        return 3./(t+1)#Sensible default
    
    def __init__(self, data, x_dim, phi, phi_dim, Y):
        self.x_dim=x_dim
        self.inputs = data[:,:-1]
        shape = list(data.shape)
        shape[-1] = 1
        self.labels = data[:,-1].reshape(shape)
        self.phi=phi
        self.label_set = Y
        self.theta_0 = zeros((phi_dim,1))
        self.phi_xy = self.phi(data)
        for x,y in zip(self.inputs,self.labels):
            self.dic_data[str(x)] = y
        print self.inputs.shape
    
    def structure(self, xy):
        return 0. if xy[-1] == self.dic_data[str(xy[:-1])] else 1.
        
    def structured_decision(self, theta):
        def decision( x ):
            score = lambda xy: dot(theta.transpose(),self.phi(xy)) + self.structure(xy)
            input_label_couples = [hstack([x,y]) for y in self.label_set]
            best_label = argmax(input_label_couples, score)[-1]
            return best_label
        vdecision = non_scalar_vectorize(decision, (self.x_dim,), (1,1))
        return lambda x: vdecision(x).reshape(x.shape[:-1]+(1,))
    
    def gradient(self, theta):
        classif_rule = self.structured_decision(theta)
        y_star = classif_rule(self.inputs)
        #print "Gradient : "+str(y_star)
        #print str(self.labels)
        phi_star = self.phi(hstack([self.inputs,y_star]))
        return mean(phi_star-self.phi_xy,axis=0)
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(StructuredClassifier,self).run( f_grad, b_norm=True)
        classif_rule = greedy_policy(theta,self.phi,self.label_set)
        return classif_rule,theta

# <codecell>

single_mu = lambda sa:feature_expectations_MC[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (2,), (729,1))
SCIRL_MC = StructuredClassifier(D_E[:,:2], 1, mu_E, 729, ACTION_SPACE)
void,reward_SCIRL_short = SCIRL_MC.run()

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
    index = s_index(state) + action*3*9*9*3
    return index

print reward_SCIRL_short.shape
reward_SCIRL=zeros((3645,1))
for state in S:
    current_indices = [sa_index(state,a) for a in ACTION_SPACE]
    reward_SCIRL[current_indices] = reward_SCIRL_short[s_index(state)]
print reward_SCIRL.shape

# <codecell>

reward_SCIRL.shape
Highway2 = MDP(P,reward_SCIRL)
mPi_A, V_A, Pi_A = Highway2.optimal_policy()
true_V_A = linalg.solve( identity( 729 ) - 0.9*dot(mPi_A,P), dot( mPi_A, R) )
Highway3 = MDP(P,rand(3645,1))
mPi_R, V_R, Pi_R = Highway3.optimal_policy()
true_V_R = linalg.solve( identity( 729 ) - 0.9*dot(mPi_R,P), dot( mPi_R, R) )
mean(true_V_A),mean(true_V_R),mean(V_E)

# <codecell>


