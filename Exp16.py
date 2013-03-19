# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#All IRL algs on the Highway
import matplotlib
matplotlib.use('Agg')
from DP import *
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *
P = genfromtxt("Highway_P.mat")
R = genfromtxt("Highway_R.mat")
Gamma_RE = 0.99
Gamma_allothers = 0.9
ACTION_SPACE = range(0,5)

# <codecell>

import sys

EPISODE_LENGTH_AND_NB_EPISODES=10
EPISODE_LENGTH_AND_NB_EPISODES=int(sys.argv[1])
RAND_STRING=str(int(rand()*10000000000))

# <codecell>

Highway = MDP(P,R)
mPi_E, V_E, Pi_E = Highway.optimal_policy()

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
l_D_E = [array(Highway.D_func(Pi_E, 1, EPISODE_LENGTH_AND_NB_EPISODES,  rho)) for i in range(0,EPISODE_LENGTH_AND_NB_EPISODES)]
D_E = vstack(l_D_E)

# <codecell>

#Relative Entropy
class GradientDescent(object):
   def alpha( self, t ):
      raise NotImplementedError, "Cannot call abstract method"

   theta_0=None
   Threshold=None
   T = -1
   sign = None
        
   def run( self, f_grad, f_proj=None, b_norm=False, b_best=True ): #grad is a function of theta
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

         if current_norm < best_norm or not b_best:
             best_norm = current_norm
             best_theta = theta.copy()
             best_iter = t
         if current_norm < self.Threshold or (self.T != -1 and t >= self.T):
             break

      print "Gradient de norme : "+str(best_norm)+", a l'iteration : "+str(best_iter)
      return best_theta
                            
class RelativeEntropy(GradientDescent):
    sign=+1.
    Threshold=0.01 #Sensible default
    T=50 #Sensible default
    Epsilon = 0.05 #RelEnt parameter, sensible default

    def alpha(self, t):
        return 1./(t+1)#Sensible default
    
    def __init__(self, mu_E, mus):
        self.theta_0 = zeros(mu_E.shape)
        self.Mu_E = mu_E
        self.Mus = mus
    
    def gradient(self, theta):
        numerator = 0
        denominator = 0
        for mu in self.Mus:
            c = exp(dot(theta.transpose(),mu))
            numerator += c*mu
            denominator += c
        assert denominator != 0,"A sum of exp(...) is null, some black magic happened here."
        return self.Mu_E - numerator/denominator - sign(theta)*self.Epsilon
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(RelativeEntropy,self).run( f_grad, b_norm=True, b_best=False)
        return theta
Mu_E = zeros((3645,1))
l_mu_E = []
for episode in l_D_E:
    mu = zeros((3645,1))
    for i in range(0,len(episode)):
        s = episode[i][0]
        a = episode[i][1]
        mu[s + a*729] += pow(Gamma_RE,i)
    mu /= float(len(episode))
    l_mu_E.append(mu)
Mu_E = mean(l_mu_E,axis=0)
Mu_E.shape
random_policy = lambda s: int(rand()*5)
Mus = []
for i in range(0,100):
    D = Highway.D_func(random_policy, 1, 10, rho)
    mu = zeros((3645,1))
    for i in range(0,10):
        s = D[i][0]
        a = D[i][1]
        mu[s + a*729] += pow(Gamma_RE,i)
    mu /= 10.
    Mus.append(mu)
Mus.append(Mu_E)
RE = RelativeEntropy(Mu_E, Mus)
reward_RE = RE.run()

Highway_RE = MDP(P,reward_RE)
mPi_RE, V_RE, Pi_RE = Highway_RE.optimal_policy()

# <codecell>

#SCIRL
#Precomputing mu with MC and heuristics
def MC_mu(episode):
    answer = zeros((729,1))
    for i in range(0,len(episode)):
        answer[episode[i]] += pow(Gamma_allothers,i)
    return answer

feature_expectations_MC = {}
d_mu_MC = {}
for episode in l_D_E:
    for start_index in range(0,len(episode)):
        data_MC=episode[start_index:,:2]
        state_action = data_MC[0,:2]
        state = data_MC[0,0]
        action = data_MC[0,1]
        mu = MC_mu(data_MC[:,0])
        try:
            d_mu_MC[str(state_action)].append(mu)
        except KeyError:
            d_mu_MC[str(state_action)] = [mu]
        for other_action in [a for a in ACTION_SPACE if a != action]:
            state_action=hstack([state,other_action])
            try:
                d_mu_MC[str(state_action)].append(Gamma_allothers*mu)
            except KeyError:
                d_mu_MC[str(state_action)]=[Gamma_allothers*mu]
for sa in d_mu_MC.keys():
    feature_expectations_MC[sa] = mean(d_mu_MC[sa],axis=0)
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
single_mu = lambda sa:feature_expectations_MC[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (2,), (729,1))
SCIRL_MC = StructuredClassifier(D_E[:,:2], 1, mu_E, 729, ACTION_SPACE)
void,reward_SCIRL_short = SCIRL_MC.run()

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
Highway_SCIRL = MDP(P,reward_SCIRL)
mPi_SCIRL, V_SCIRL, Pi_SCIRL = Highway_SCIRL.optimal_policy()

# <codecell>

#CSI
#Classification

class BlockOptimizedStructuredClassifier(GradientDescent):
    sign=-1.
    Threshold=0.01 #Sensible default
    T=100 #Sensible default
    phi=None
    label_set=None
    
    def alpha(self, t):
        return 3./(t+1)#Sensible default
    
    def __init__(self, psi, actions, nb_actions):
        self.label_set=range(0,nb_actions)
        self.N,self.K = psi.shape
        self.A = nb_actions
        self.theta_0 = zeros(self.K*self.A)
        self.ExpertDecision = zeros((self.N,self.A))
        for i,j in zip( range(0,self.N), actions.reshape(self.N) ):
            self.ExpertDecision[i,j] = 1.
        self.Structure = array(self.ExpertDecision!=1)
        self.ExpertDecision = self.ExpertDecision.reshape(self.N,self.A,1)
        self.Psi_3 = array([[p for i in range(0,self.A)] for p in psi])
        self.Phi = self.ExpertDecision*self.Psi_3
        self.Phi = self.Phi.reshape(self.N,self.K*self.A)
        self.Psi = psi
    
    def gradient(self, theta):
        theta_2 = hstack([theta[i*self.K:(i+1)*self.K].reshape((self.K,1)) for i in range(0,self.A)])
        score = dot(self.Psi,theta_2)+self.Structure
        maxScore = dot(score.max(axis=1).reshape((self.N,1)),ones((1,self.A)))
        decision = (score==maxScore).reshape(self.N,self.A,1)
        #We restrict ourselves to one arbitrary decision
        for i in range(0,self.N):
            gotOne = False
            for j in range(0,self.A):
                if decision[i,j] and not gotOne:
                    gotOne = True
                elif decision[i,j] and gotOne:
                    decision[i,j] = False
        phi_star = decision*self.Psi_3
        phi_star = phi_star.reshape(self.N,self.K*self.A)
        return mean(phi_star-self.Phi,axis=0)
    
    def run(self):
        f_grad = lambda theta: self.gradient(theta)
        theta = super(BlockOptimizedStructuredClassifier,self).run( f_grad, b_norm=True)
        return theta

Psi = zeros((len(D_E),729))
for i,j in zip(range(0,len(D_E)),D_E[:,0]):
    Psi[i,j]=1.
clf = BlockOptimizedStructuredClassifier(Psi, D_E[:,1], 5)
theta_C = clf.run()
mPi_C = Highway.Q2Pi(theta_C)

def single_phi(sa):
    answer = zeros((3645,1))
    answer[sa[0] + 729*sa[1]] = 1.
    return answer
phi = non_scalar_vectorize(single_phi, (2,), (3645,1))
q = lambda sa: squeeze(dot(theta_C.transpose(),phi(sa)))
#pi_c = greedy_policy(theta_C,phi,ACTION_SPACE, s_dim=1)
single_pi_c = lambda s : Highway.control(s, mPi_C)
pi_c = vectorize(single_pi_c)
#Données pour la regression
column_shape = (len(D_E),1)
s = D_E[:,0].reshape(column_shape)
a = D_E[:,1].reshape(column_shape)
sa = D_E[:,:2]
s_dash = D_E[:,3].reshape(column_shape)
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-Gamma_allothers*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
##Avec l'heuristique : 
thetas = zeros((5,729))
for action in ACTION_SPACE:
    X = Psi
    hat_r_bool_table = array(a==action)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    Y =  hat_r_bool_table*hat_r + r_min_bool_table*r_min
    #if sum(hat_r_bool_table) == 0:
    #    thetas[action] = zeros(729)
    #else:
    thetas[action] = squeeze( dot(dot(inv(dot(X.transpose(),X)+0.1*identity(X.shape[1])),X.transpose()),Y) )
theta_CSI = thetas.reshape(3645)
fCSI_reward = lambda sa:dot(theta_CSI.transpose(),phi(sa))
reward_CSI=zeros((3645,1))
for state in S:
    for action in ACTION_SPACE:
        index = sa_index(state, action)
        reward_CSI[index] = fCSI_reward(array([s_index(state),action]))
Highway_CSI = MDP(P,reward_CSI)
mPi_CSI, V_CSI, Pi_CSI = Highway_CSI.optimal_policy()

# <codecell>

Highway_R = MDP(P,rand(3645,1))
mPi_R, V_R, Pi_R = Highway_R.optimal_policy()

# <codecell>

print Highway.evaluate(mPi_R),Highway.evaluate(mPi_C),Highway.evaluate(mPi_RE),Highway.evaluate(mPi_SCIRL),Highway.evaluate(mPi_CSI)

