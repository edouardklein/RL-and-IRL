# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from stuff import *
from pylab import *
from random import *
import numpy
from rl import *

# <codecell>

Psi = genfromtxt('asterix/psi.mat')
A = genfromtxt('asterix/actions.mat')
A = A.reshape((len(A),1))
ACTION_SPACE = range(0,18)
Psi.shape,A.shape

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
    T=10 #Sensible default
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

#Running the classifier gives pi_c and q
def asterix_single_phi(psi_a):
    psi = psi_a[:1000]
    a = psi_a[-1]
    answer = zeros((18000,1))
    index = a*1000
    answer[index:index+1000,0] = psi
    return answer
asterix_phi = non_scalar_vectorize(asterix_single_phi, (1001,), (18000,1))
clf = StructuredClassifier(hstack([Psi,A]), 1000, asterix_phi, 18000, ACTION_SPACE)
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

