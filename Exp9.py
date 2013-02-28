# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

# <codecell>

#Classification code
def greedy_policy( omega, phi, A ): 
    def policy( *args ):
        state_actions = [hstack(args+(a,)) for a in A]
        q_value = lambda sa: float(dot(omega.transpose(),phi(sa)))
        best_action = argmax( state_actions, q_value )[-1] #FIXME6: does not work for multi dimensional actions
        return best_action
    vpolicy = non_scalar_vectorize( policy, (1000,), (1,1) ) #FIXME, the 1000 here is s_dim and is problem-dependant
    return lambda state: vpolicy(state).reshape(state.shape[:-1]+(1,))

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
    
    def __init__(self, psi, actions, nb_actions):
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
        classif_rule = greedy_policy(theta,self.phi,self.label_set)
        return classif_rule,theta

# <codecell>

Psi = genfromtxt('asterix/psi.mat')
A = genfromtxt('asterix/actions.mat')
A = A.reshape((len(A),1))
ACTION_SPACE = range(0,18)
Psi.shape,A.shape

# <codecell>

#Running the classifier gives pi_c and q
clf = StructuredClassifier(Psi, A, 18)
pi_c,theta_C = clf.run()
q = lambda sa: squeeze(dot(theta_C.transpose(),asterix_phi(sa)))
savetxt("CSI_asterix_theta_C.mat", theta_C)

# <codecell>

#Regressor code
q = lambda sa: squeeze(dot(theta_C.transpose(),asterix_phi(sa)))

# <codecell>

#Running the regressio code yelds 18 theta matrices, one per action
column_shape = (len(Psi)-1,1)
s = Psi[:-1,:]
s.shape
a = A[:-1].reshape(column_shape)
sa = hstack([s,a])
s_dash = Psi[1:,:]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)

# <codecell>

thetas = zeros((18,1000))
for action in ACTION_SPACE:
    X = s
    hat_r_bool_table = array(a==action)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    Y =  hat_r_bool_table*hat_r + r_min_bool_table*r_min
    thetas[action] = squeeze( dot(dot(inv(dot(X.transpose(),X)+0.1*identity(X.shape[1])),X.transpose()),Y) )
    

# <codecell>

savetxt("CSI_asterix_thetas.mat", thetas)

