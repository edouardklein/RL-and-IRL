# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#SCIRLBoost on the Highway
from DP import *
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *
mP = genfromtxt("Highway_P.mat")
R = genfromtxt("Highway_R.mat")
Gamma = 0.9
ACTION_SPACE = range(0,5)
Highway = MDP(mP,R)
mPi_E, V_E, Pi_E = Highway.optimal_policy()

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
         #print "Norme du gradient : "+str(current_norm)+", pas : "+str(self.alpha(t))+", iteration : "+str(t)

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
    Threshold=0.01 #Sensible default
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

rho = lambda : int(rand()*729) #uniform distribtion over S
l_D_E = [array(Highway.D_func(Pi_E, 1, 7,  rho)) for i in range(0,7)]
D_E = vstack(l_D_E)
N = len(D_E)
D_sasa = D_E.copy()
for i in range(0,N):
    D_sasa[i,3] = Pi_E(D_E[i,2]) #Making Sasa from sasr
D_sasa

# <codecell>

dic_Psi = {}
for s in D_sasa[:,0]:
    if not (str(s) in dic_Psi):
        dic_Psi[str(squeeze(s))] = rand()
f_Psi = lambda s: dic_Psi[str(squeeze(s))]
Psi = zeros((N,1))
for i in range(0,N):
    Psi[i] = f_Psi(D_sasa[i,0])
Psi

# <codecell>

P = 1
theta = zeros((P,1))

# <codecell>

#while True: #Do..while loop
#Precomputing mu with MC and heuristics
def MC_mu(episode):
    answer = zeros((P,1))
    for i in range(0,len(episode)):
        answer += pow(Gamma,i) * f_Psi(episode[i])
    return answer/float(len(episode))

feature_expectations_MC = {}
d_mu_MC = {}
for episode in l_D_E:
    for start_index in range(0,len(episode)):
        data_MC=episode[start_index:,:2]
        state_action = data_MC[0,:2]
        state = data_MC[0,0]
        action = data_MC[0,1]
        mu = MC_mu(data_MC[:,0])
        print state_action
        try:
            d_mu_MC[str(state_action)].append(mu)
        except KeyError:
            d_mu_MC[str(state_action)] = [mu]
        for other_action in [a for a in ACTION_SPACE if a != action]:
            state_action=hstack([state,other_action])
            try:
                d_mu_MC[str(state_action)].append(Gamma*mu)
            except KeyError:
                d_mu_MC[str(state_action)]=[Gamma*mu]
for sa in d_mu_MC.keys():
    feature_expectations_MC[sa] = mean(d_mu_MC[sa],axis=0)
#feature_expectations_MC

# <codecell>

single_mu = lambda sa:feature_expectations_MC[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (2,), (P,1))
SCIRL_MC = StructuredClassifier(D_E[:,:2], 1, mu_E, P, ACTION_SPACE)
void,theta = SCIRL_MC.run()

# <codecell>

pi = SCIRL_MC.structured_decision(theta)
a_star = pi(D_sasa[:,0].reshape((N,1)))
bool_star = (D_sasa[:,-1].reshape((N,1)) != a_star)
D_diff = D_sasa[bool_star]
print "|D_diff| = "+str(len(D_diff))

# <codecell>

A = zeros((p,N))
b = zeros((N,p))

for i in range(0,N):
    sa = D_sasa[i,:2]
    b[i] = feature_expectations_MC[str(sa)].transpose()

for line in D_sasa:
    sa = line[:2]
    sa_dash = line[-2:]
    mu = feature_expectations_MC[str(sa)]
    mu_dash = feature_expectations_MC[str(sa_dash)]
    A += dot(inv(dot(mu,(mu-Gamma*mu_dash).transpose())),b.transpose())
print A
    

# <codecell>

print l_D_E[0]

