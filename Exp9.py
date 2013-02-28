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
def asterix_single_phi(psi_a):
    psi = psi_a[:1000]
    a = psi_a[-1]
    answer = zeros((18000,1))
    index = a*1000
    answer[index:index+1000,0] = psi
    return answer
asterix_phi = non_scalar_vectorize(asterix_single_phi, (1001,), (18000,1))



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
    T=100 #Sensible default
    phi=asterix_phi
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

# <codecell>

Psi = genfromtxt('asterix/psi2.mat')
A = genfromtxt('asterix/actions2.mat')
A = A.reshape((len(A),1))
ACTION_SPACE = range(0,18)
Psi.shape,A.shape

# <codecell>

#Running the classifier gives pi_c and q
clf = StructuredClassifier(Psi, A, 18)
theta_C = clf.run()
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
a_dash = -ones(column_shape)
#a_dash = pi_c(s_dash).reshape(column_shape)
theta_2 = hstack([theta_C[i*1000:(i+1)*1000].reshape((1000,1)) for i in range(0,18)])
score = dot(s_dash,theta_2)
maxScore = dot(score.max(axis=1).reshape((len(Psi)-1,1)),ones((1,18)))
decision = (score==maxScore).reshape(len(Psi)-1,18,1)
for i in range(0,len(Psi)-1):
    gotOne = False
    for j in range(0,18):
        if decision[i,j] and not gotOne:
            gotOne = True
            a_dash[i] = j
        elif decision[i,j] and gotOne:
            decision[i,j] = False
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
    if sum(hat_r_bool_table) == 0:
        thetas[action] = zeros(1000)
    else:
        thetas[action] = squeeze( dot(dot(inv(dot(X.transpose(),X)+0.1*identity(X.shape[1])),X.transpose()),Y) )
    

# <codecell>

savetxt("CSI_asterix_thetas.mat", thetas)

