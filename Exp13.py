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

# <codecell>

rho = lambda : int(rand()*729) #uniform distribtion over S
D_E = array(Highway.D_func(Pi_E, 1, 100,  rho))

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
    Threshold=0.1 #Sensible default
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
def single_phi(sa):
    answer = zeros((3645,1))
    answer[sa[0] + 729*sa[1]] = 1.
    return answer
phi = non_scalar_vectorize(single_phi, (2,), (3645,1))
q = lambda sa: squeeze(dot(theta_C.transpose(),phi(sa)))
pi_c = greedy_policy(theta_C,phi,ACTION_SPACE, s_dim=1)

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

##Avec l'heuristique : 
regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in ACTION_SPACE] 
def add_output_column( reg_mat ):
    actions = reg_mat[:,-1].reshape(column_shape)
    hat_r_bool_table = array(actions==a)
    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
    return hstack([reg_mat,output_column])
regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#Régression
from sklearn.svm import SVR
y = regression_matrix[:,-1]
X = regression_matrix[:,:-1]
reg = SVR(C=1.0, epsilon=0.2)#, gamma=1/(2*pow(0.03,2)))
reg.fit(X, y)
fCSI_reward = lambda sa:reg.predict(sa)[0]

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
    index = s_index(state) + a*3*9*9*3
    return index

reward_CSI=zeros((3645,1))
for state in S:
    for action in ACTION_SPACE:
        index = sa_index(state, action)
        reward_CSI[index] = fCSI_reward(array([state,action]))

# <codecell>

Highway2 = MDP(P,reward_CSI)
mPi_A, V_A, Pi_A = Highway2.optimal_policy()
true_V_A = linalg.solve( identity( 729 ) - 0.9*dot(mPi_A,P), dot( mPi_A, R) )
Highway3 = MDP(P,rand(3645,1))
mPi_R, V_R, Pi_R = Highway3.optimal_policy()
true_V_R = linalg.solve( identity( 729 ) - 0.9*dot(mPi_R,P), dot( mPi_R, R) )
mean(true_V_A),mean(true_V_R),mean(V_E)

