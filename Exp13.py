# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

#CSI on the Highway
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
Highway.evaluate(mPi_E)

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
l_D_E = [array(Highway.D_func(Pi_E, 1, 7,  rho)) for i in range(0,7)]
D_E = vstack(l_D_E)

# <codecell>

#Classification
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
        theta = super(StructuredClassifier,self).run( f_grad, b_norm=True)
        return theta
    
Psi = zeros((len(D_E),729))
for i,j in zip(range(0,len(D_E)),D_E[:,0]):
    Psi[i,j]=1.
clf = StructuredClassifier(Psi, D_E[:,1], 5)
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
Highway.evaluate(mPi_C)
#theta_C.min(),theta_C.max(),theta_C.mean(),theta_C.var()

# <codecell>

#Données pour la regression
column_shape = (len(D_E),1)
s = D_E[:,0].reshape(column_shape)
a = D_E[:,1].reshape(column_shape)
sa = D_E[:,:2]
s_dash = D_E[:,3].reshape(column_shape)
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-Gamma*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
hat_r.max(),hat_r.min(),hat_r.mean(),hat_r.var()

# <codecell>

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
theta_CSI.min(),theta_CSI.max(),theta_CSI.mean(),theta_CSI.var()

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

reward_CSI=zeros((3645,1))
for state in S:
    for action in ACTION_SPACE:
        index = sa_index(state, action)
        reward_CSI[index] = fCSI_reward(array([s_index(state),action]))
Highway2 = MDP(P,reward_CSI)
mPi_A, V_A, Pi_A = Highway2.optimal_policy()
Highway.evaluate(mPi_A)
#reward_CSI.min(),reward_CSI.max(),reward_CSI.mean(),reward_CSI.var()

# <codecell>


Highway3 = MDP(P,rand(3645,1))
mPi_R, V_R, Pi_R = Highway3.optimal_policy()
Highway.evaluate(mPi_R)

# <codecell>

savetxt('RewardFaisantPlanterDP.mat',reward_CSI)

