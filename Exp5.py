# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#SCIRL mountain car
#Mountain Car
from stuff import *
from pylab import *
from random import *
import numpy
from rl import *
import sys
ACTION_SPACE=[-1,0,1]
NB_SAMPLES=10
NB_SAMPLES=int(sys.argv[1])
RAND_STRING=str(int(rand()*1000000))
def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    next_speed = next_speed if next_speed > -0.07 else -0.07
    next_speed = next_speed if next_speed < 0.07 else 0.07
    next_position = next_position if next_position > -1.2 else -1.2
    next_position = next_position if next_position < 0.6 else 0.6
    return array([next_position,next_speed])
def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])
def mountain_car_interesting_state():
    position = choice([numpy.random.uniform(low=-1.2,high=-0.8),numpy.random.uniform(low=0,high=0.6)])
    speed = choice([numpy.random.uniform(low=-0.07,high=-0.04),numpy.random.uniform(low=0.04,high=0.07)])
    if position < 0 and speed < 0:
        speed=-speed
    return array([position,speed])
mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,5),linspace(-0.07,0.07,5))

mountain_car_sigma_position = 2*pow((0.6+1.2)/5.,2)
mountain_car_sigma_speed = 2*pow((0.07+0.07)/5.,2)

def mountain_car_single_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(5*5,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position 
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    return array(psi).reshape((5*5,1))
mountain_car_psi= non_scalar_vectorize(mountain_car_single_psi,(2,),(25,1))
def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros((5*5*3,1))
    answer[index_action*5*5:index_action*5*5+5*5] = mountain_car_single_psi(state)
    return answer

mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(75,1))

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0

def mountain_car_training_data(freward=mountain_car_reward):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,500):
        state = mountain_car_uniform_state()
        for i in range(0,10):
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)
#omega=genfromtxt("mountain_car_expert_omega.mat")
#policy=greedy_policy(omega, mountain_car_phi, ACTION_SPACE)
data = mountain_car_training_data()
policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )
savetxt("data/Expert_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega)
def mountain_car_testing_data(policy):
    traj = []
    state = mountain_car_interesting_state()
    t=0
    reward = 0
    while t < 300 and reward == 0:
        t+=1
        action = policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=mountain_car_interesting_state()
    return array(traj)
while True:#Do..while
    data = vstack([mountain_car_testing_data(policy) for i in range(0,10)])
    TRANS = array([choice(data) for i in range(0,NB_SAMPLES)])
    actions = TRANS[:,2]
    if any(actions == 0) and any(actions == -1) and any(actions == 1):
        break
psi=mountain_car_psi
phi=mountain_car_phi
s=TRANS[:,:2]
a=TRANS[:,2]
s_dash=TRANS[:,3:5]
a_dash=TRANS[:,5]
sa=TRANS[:,:3]
sa_dash=TRANS[:,3:6]

##SCIRL
#Precomputing mu, with Monte-Carlo + heuristics
#On est obligés d'utiliser LSTDmu puisqu'on a que des transitions décorellées et non des trajectoires
A = zeros((75,75))
b = zeros((75,25))
phi_t = phi(sa)
phi_t_dash = phi(sa_dash)
psi_t = psi(s)
for phi_t,phi_t_dash,psi_t in zip(phi_t,phi_t_dash,psi_t):
    A = A + dot(phi_t,
            (phi_t - GAMMA*phi_t_dash).transpose())
    b = b + dot(phi_t,psi_t.transpose())
omega_lstd_mu = dot(inv(A+LAMBDA*identity(75)),b)
phi_t.shape, phi_t_dash.shape, psi_t.shape
feature_expectations = {}
for state,action in zip(s,a):
    state_action = hstack([state,action])
    mu = dot(omega_lstd_mu.transpose(),phi(state_action))
    feature_expectations[str(state_action)] = mu
    for other_action in [a for a in ACTION_SPACE if a != action]:
        state_action=hstack([state,other_action])
        feature_expectations[str(state_action)]=GAMMA*mu
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
         if norm < self.Threshold or (self.T != -1 and t >= self.T):
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
single_mu = lambda sa:feature_expectations[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (3,), (25,1))
SCIRL = StructuredClassifier(sa, 2, mu_E, 25, ACTION_SPACE)
void,theta_SCIRL = SCIRL.run()
#Evaluation de SCIRL
SCIRL_reward = lambda sas:dot(theta_SCIRL.transpose(),psi(sas[:2]))[0]
vSCIRL_reward = non_scalar_vectorize( SCIRL_reward, (5,),(1,1) )
data_LSPI = mountain_car_training_data(freward=SCIRL_reward)
policy_CSI,omega_CSI = lspi( data_LSPI, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )#None,zeros((75,1))#
savetxt("data/SCIRL_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_CSI)
s=TRANS[:,:2]
a=TRANS[:,2]
s_dash=TRANS[:,3:5]
a_dash=TRANS[:,5]
sa=TRANS[:,:3]
sa_dash=TRANS[:,3:6]

##CSI
#Classification
from sklearn import svm
clf = svm.SVC(C=10, probability=True)
clf.fit(s, a)
import pickle
RAND_STRING=str(int(rand()*100000))
with open('data/Classif_'+str(NB_SAMPLES)+"_"+RAND_STRING+".obj", 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
clf_predict= lambda state : clf.predict(squeeze(state))
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[-1]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
#Données pour la regression
column_shape = (len(TRANS),1)
s = TRANS[:,0:2]
a = TRANS[:,2].reshape(column_shape)
sa = TRANS[:,0:3]
s_dash = TRANS[:,3:5]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-GAMMA*q(sa_dash)).reshape(column_shape)
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
reg = SVR(C=1.0, epsilon=0.2)
reg.fit(X, y)
CSI_reward = lambda sas:reg.predict(sas[:3])[0]
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
#Evaluation de CSI
data_LSPI = mountain_car_training_data(freward=CSI_reward)
policy_CSI,omega_CSI = lspi( data_LSPI, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=75, iterations_max=20 )#None,zeros((75,1))#
savetxt("data/CSI_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_CSI)
def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 1500 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_tricky_episode_length(policy):
    return mountain_car_episode_length(-0.9,0,policy)

print "Nb_samples :\t"+str(NB_SAMPLES)
print "Length CSI :\t"+str(mountain_car_tricky_episode_length(policy_CSI))
print "Length Class :\t"+str(mountain_car_tricky_episode_length(pi_c))
print "Length exp :\t"+str(mountain_car_tricky_episode_length(policy))

