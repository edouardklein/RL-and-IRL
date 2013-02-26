# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Mountain Car
from stuff import *
from pylab import *
from random import *
import pickle
import numpy
from rl import *
import sys

NB_SAMPLES=100
NB_SAMPLES=int(sys.argv[1])
RAND_STRING=str(int(rand()*10000000000))


ACTION_SPACE=[-1,0,1]



def mountain_car_next_state(state,action):
    position,speed=state
    next_speed = squeeze(speed+action*0.001+cos(3*position)*(-0.0025))
    next_position = squeeze(position+next_speed)
    if not -0.07 <= next_speed <= 0.07:
        next_speed = sign(next_speed)*0.07
    if not -1.2 <= next_position <= 0.6:
        next_speed=0.
        next_position = -1.2 if next_position < -1.2 else 0.6
    return array([next_position,next_speed])

def mountain_car_uniform_state():
    return array([numpy.random.uniform(low=-1.2,high=0.6),numpy.random.uniform(low=-0.07,high=0.07)])

mountain_car_mu_position, mountain_car_mu_speed = meshgrid(linspace(-1.2,0.6,7),linspace(-0.07,0.07,7))

mountain_car_sigma_position = 2*pow((0.6+1.2)/10.,2)
mountain_car_sigma_speed = 2*pow((0.07+0.07)/10.,2)

def mountain_car_single_psi(state):
    position,speed=state
    psi=[]
    for mu in zip_stack(mountain_car_mu_position, mountain_car_mu_speed).reshape(7*7,2):
        psi.append(exp( -pow(position-mu[0],2)/mountain_car_sigma_position 
                        -pow(speed-mu[1],2)/mountain_car_sigma_speed))
    psi.append(1.)
    return array(psi).reshape((7*7+1,1))

mountain_car_psi= non_scalar_vectorize(mountain_car_single_psi,(2,),(50,1))

def mountain_car_single_phi(sa):
    state=sa[:2]
    index_action = int(sa[-1])+1
    answer=zeros(((7*7+1)*3,1))
    answer[index_action*(7*7+1):index_action*(7*7+1)+7*7+1] = mountain_car_psi(state)
    return answer

mountain_car_phi= non_scalar_vectorize(mountain_car_single_phi,(3,),(150,1))

def mountain_car_reward(sas):
    position=sas[0]
    return 1 if position > 0.5 else 0

def mountain_car_episode_length(initial_position,initial_speed,policy):
    answer = 0
    reward = 0.
    state = array([initial_position,initial_speed])
    while answer < 300 and reward == 0. :
        action = policy(state)
        next_state = mountain_car_next_state(state,action)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        state=next_state
        answer+=1
    return answer

def mountain_car_episode_vlength(policy):
    return vectorize(lambda p,s:mountain_car_episode_length(p,s,policy))


def mountain_car_training_data(freward=mountain_car_reward,traj_length=5,nb_traj=1000):
    traj = []
    random_policy = lambda s:choice(ACTION_SPACE)
    for i in range(0,nb_traj):
        state = mountain_car_uniform_state()
        reward=0
        t=0
        while t < traj_length and reward == 0:
            t+=1
            action = random_policy(state)
            next_state = mountain_car_next_state(state, action)
            reward = freward(hstack([state, action, next_state]))
            traj.append(hstack([state, action, next_state, reward]))
            state=next_state
    return array(traj)

def mountain_car_manual_policy(state):
    position,speed = state
    return -1. if speed <=0 else 1.

def mountain_car_interesting_state():
    position = numpy.random.uniform(low=-1.2,high=-0.9)
    speed = numpy.random.uniform(low=-0.07,high=0)
    return array([position,speed])

def mountain_car_IRL_traj():
    traj = []
    state = mountain_car_interesting_state()
    reward = 0
    while reward == 0:
        action = mountain_car_manual_policy(state)
        next_state = mountain_car_next_state(state, action)
        next_action = mountain_car_manual_policy(next_state)
        reward = mountain_car_reward(hstack([state, action, next_state]))
        traj.append(hstack([state, action, next_state, next_action, reward]))
        state=next_state
    return array(traj)

def mountain_car_IRL_data(nbsamples):
    data = mountain_car_IRL_traj()
    while len(data) < nbsamples:
        data = vstack([data,mountain_car_IRL_traj()])
    return data[:nbsamples]

TRAJS = mountain_car_IRL_data(NB_SAMPLES)

psi=mountain_car_psi
phi=mountain_car_phi
s=TRAJS[:,:2]
a=TRAJS[:,2]
#Classification
from sklearn import svm
clf = svm.SVC(C=1, probability=True, gamma=1/(2*pow(0.03,2)))
clf.fit(s, a)
def clf_predict(state):
    try:
        return clf.predict(squeeze(state))
    except ValueError:
        return 1.
vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
def clf_score(sa):
    #try:
    action = sa[-1]
    index=0
    if action == -1.:
        index = 0
    elif action == 1.:
        index = 1
    else:
        return 0
    return squeeze(clf.predict_proba(squeeze(sa[:2])))[sa[index]]
vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
#Données pour la regression
column_shape = (len(TRAJS),1)
s = TRAJS[:,0:2]
a = TRAJS[:,2].reshape(column_shape)
sa = TRAJS[:,0:3]
s_dash = TRAJS[:,3:5]
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
reg = SVR(C=1.0, epsilon=0.2, gamma=1/(2*pow(0.03,2)))
reg.fit(X, y)
CSI_reward = lambda sas:reg.predict(sas[:3])[0]
vCSI_reward = non_scalar_vectorize( CSI_reward, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vCSI_reward(data[:,:5]))
policy_CSI,omega_CSI = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )
def mountain_car_testing_state():
    position = numpy.random.uniform(low=-1.2,high=0.5)
    speed = numpy.random.uniform(low=-0.07,high=0.07)
    return array([position,speed])

def mountain_car_mean_performance(policy):
    return mean([mountain_car_episode_length(state[0],state[1],policy) for state in [mountain_car_testing_state() for i in range(0,100)]])
print "Samples : "+str(NB_SAMPLES)
print "CSI, classif : "
print mountain_car_mean_performance(policy_CSI),mountain_car_mean_performance(pi_c)
savetxt("data/CSI_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_CSI)
with open('data/Classif_'+str(NB_SAMPLES)+"_"+RAND_STRING+".obj", 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)


# <codecell>

psi=mountain_car_psi
phi=mountain_car_phi
s=TRAJS[:,:2]
a=TRAJS[:,2]

s_dash=TRAJS[:,3:5]
a_dash=TRAJS[:,5]
sa=TRAJS[:,:3]
sa_dash=TRAJS[:,3:6]

##SCIRL
#Precomputing mu with LSTDmu and heuristics
A = zeros((150,150))
b = zeros((150,50))
phi_t = phi(sa)
phi_t_dash = phi(sa_dash)
psi_t = psi(s)
for phi_t,phi_t_dash,psi_t in zip(phi_t,phi_t_dash,psi_t):
    A = A + dot(phi_t,
            (phi_t - GAMMA*phi_t_dash).transpose())
    b = b + dot(phi_t,psi_t.transpose())
omega_lstd_mu = dot(inv(A+0.1*identity(150)),b)
phi_t.shape, phi_t_dash.shape, psi_t.shape
feature_expectations = {}
for state,action in zip(s,a):
    state_action = hstack([state,action])
    mu = dot(omega_lstd_mu.transpose(),phi(state_action))
    feature_expectations[str(state_action)] = mu
    for other_action in [a for a in ACTION_SPACE if a != action]:
        state_action=hstack([state,other_action])
        feature_expectations[str(state_action)]=GAMMA*mu
        
        
#Precomputing mu with MC and heuristics
feature_expectations_MC = {}
for start_index in range(0,len(TRAJS)):
    end_index = (i for i in range(start_index,len(TRAJS)) if TRAJS[i,6] == 1 or i==len(TRAJS)-1).next()
    #print "start_index : "+str(start_index)+" end_index : "+str(end_index)
    data_MC=TRAJS[start_index:end_index+1,:3]
    GAMMAS = range(0,len(data_MC))
    GAMMAS = array(map( lambda x: pow(GAMMA,x), GAMMAS))
    state_action = data_MC[0,:3]
    state = data_MC[0,:2]
    action = data_MC[0,2]
    mu = None
    if len(data_MC) > 1:
        mu = dot( GAMMAS,squeeze(psi(data_MC[:,:2])))
    else:
        mu = squeeze(psi(squeeze(data_MC[:,:2])))
    feature_expectations_MC[str(state_action)] = mu
    for other_action in [a for a in ACTION_SPACE if a != action]:
        state_action=hstack([state,other_action])
        feature_expectations_MC[str(state_action)]=GAMMA*mu
        

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

#Version LSTDmu    
single_mu = lambda sa:feature_expectations[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (3,), (50,1))
SCIRL = StructuredClassifier(sa, 2, mu_E, 50, ACTION_SPACE)
void,theta_SCIRL = SCIRL.run()
#Evaluation de SCIRL
SCIRL_reward = lambda sas:dot(theta_SCIRL.transpose(),psi(sas[:2]))[0]
vSCIRL_reward = non_scalar_vectorize( SCIRL_reward, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vSCIRL_reward(data[:,:5]))
policy_SCIRL,omega_SCIRL = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )#None,zeros((75,1))#
savetxt("data/SCIRL_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_SCIRL)
#Version MC_mu
single_mu = lambda sa:feature_expectations_MC[str(sa)]
mu_E = non_scalar_vectorize(single_mu, (3,), (50,1))
SCIRL_MC = StructuredClassifier(sa, 2, mu_E, 50, ACTION_SPACE)
void,theta_SCIRL_MC = SCIRL_MC.run()
#Evaluation de SCIRL
SCIRL_reward = lambda sas:dot(theta_SCIRL_MC.transpose(),psi(sas[:2]))[0]
vSCIRL_reward = non_scalar_vectorize( SCIRL_reward, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vSCIRL_reward(data[:,:5]))
policy_SCIRL_MC,omega_SCIRL_MC = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )#None,zeros((75,1))#
savetxt("data/SCIRLMC_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_SCIRL_MC)
print "SCIRL, SCIRL MC : "
print mountain_car_mean_performance(policy_SCIRL),mountain_car_mean_performance(policy_SCIRL_MC)

# <codecell>

GAMMA = 0.99

def end_of_episode(data,i):
    try:
        if all(data[i,3:5] == data[i+1,:2]):
            return False
        else:
            return True
    except:
        return True

#function reward=relative_entropy_boularias(phi,n_s,gamma,data_c,data_r,L_c,H_c,L_r,H_r,delta,epsilon_re,N_final)
def relative_entropy(data_c,data_r,delta):
    #size_phi=size(phi);
    
    #feature_c=zeros(L_c,size_phi(2));
    #feature_r=zeros(L_r,size_phi(2));
    feature_c=[]
    feature_r=[]
    
    #theta=zeros(1,size_phi(2));
    theta=zeros((1,150))
    #for i=1:L_c
        #for j=1:H_c
    eoe_indices = [i for i in range(0,len(data_c)) if end_of_episode(data_c,i)]
    for start_index,end_index in zip( [0] + map(lambda x:x+1,eoe_indices[:-1]),eoe_indices):
        #feature_c(i,:)=feature_c(i,:)+gamma^j*phi(data_c(j+H_c*(i-1),1)+n_s*(data_c(j+H_c*(i-1),2)-1),:);
        data_MC=data_c[start_index:end_index+1,:3]
        GAMMAS = range(0,len(data_MC))
        GAMMAS = array(map( lambda x: pow(GAMMA,x), GAMMAS))
        state_action = data_MC[0,:3]
        mu = None
        if len(data_MC) > 1:
            mu = dot( GAMMAS,squeeze(phi(data_MC[:,:3])))
        else:
            mu = squeeze(phi(squeeze(data_MC[:,:3])))
        feature_c.append(mu)   
    feature_c=array(feature_c)

    #feature_c_mean=mean(feature_c,1);
    feature_c_mean=mean(feature_c,0);
    
    #epsilon=sqrt(-log2(1-delta)/(2*H_c))*(gamma^(H_c+1)-1)/(gamma-1);
    #epsilon=sqrt(-log2(1-delta)/(2*300))*(pow(GAMMA,(300+1))-1)/(GAMMA-1);
    epsilon = 0.01
    print "epsilon "+str(epsilon)
    
    #for i=1:L_r
    #    for j=1:H_r
    eoe_indices = [i for i in range(0,len(data_r)) if end_of_episode(data_r,i)]
    for start_index,end_index in zip( [0] + map(lambda x:x+1,eoe_indices[:-1]),eoe_indices):
        #feature_r(i,:)=feature_r(i,:)+gamma^j*phi(data_r(j+H_r*(i-1),1)+n_s*(data_r(j+H_r*(i-1),2)-1),:)
        data_MC=data_r[start_index:end_index+1,:3]
        GAMMAS = range(0,len(data_MC))
        GAMMAS = array(map( lambda x: pow(GAMMA,x), GAMMAS))
        state_action = data_MC[0,:3]
        mu = None
        if len(data_MC) > 1:
            mu = dot( GAMMAS,squeeze(phi(data_MC[:,:3])))
        else:
            mu = squeeze(phi(squeeze(data_MC[:,:3])))
        feature_r.append(mu)
        #    end   
        #end
    feature_r=array(feature_r)

    #Criterion=epsilon_re+1;
    criterion=numpy.inf
    #counter=1;
    counter=1

    #while Criterion>epsilon_re&&counter<N_final
    while criterion > 0.01 and counter < 100:
        #buffer_derivative_t=zeros(1,size_phi(2));
        derivative = zeros((1,150))
        #buffer_t=0;
        t = 0
        #buffer_theta=theta;
        #for i=1:L_r
        for i in range(0,len(feature_r)):
            #buffer_derivative_t=buffer_derivative_t+exp(theta*feature_r(i,:)')*feature_r(i,:);
            derivative += exp(dot(theta,feature_r[i,:].transpose()))*feature_r[i,:]
            #buffer_t=buffer_t+exp(theta*feature_r(i,:)');
            t +=  exp(dot(theta,feature_r[i,:].transpose()))
            #end    

        #buffer_derivative=feature_c_mean-buffer_derivative_t/(buffer_t)-sign(theta)*epsilon;
        derivative=feature_c_mean-derivative/(t)-sign(theta)*epsilon

        print "Boubou run \t"+str(counter)+" criterion is \t"+str(criterion)+" ||derivative|| is \t"+str(norm(derivative))
        if isnan(derivative):
            raise Exception
        
        #if norm(buffer_derivative)==0
        if norm(derivative)==0:
            #theta=buffer_theta;
            break
        else:
            #theta=buffer_theta+(1/counter)*buffer_derivative/norm(buffer_derivative,2);
            delta_theta = (100./float(counter))*derivative/norm(derivative,2)
            #end
        #Criterion=norm(buffer_theta-theta,2);
        criterion=norm(delta_theta,2)
        theta+=delta_theta
        #counter=counter+1;    
        counter += 1
        #end    
    print "Stop boubou @ run "+str(counter-1)+" criterion is "+str(criterion)
    #reward=phi*theta';
    return lambda sas: dot(theta,phi(sas[:3]))[0]
    

#data_r = genfromtxt("mountain_car_batch_data.mat")
data_r = genfromtxt("mountain_car_boubou_trajs.mat")

toto = True
while toto:
    try:
        RE_reward = relative_entropy(TRAJS, data_r, 0.99)
        print "Trying again "
        toto = False
    except Exception:
        pass
vRE_reward = non_scalar_vectorize( RE_reward, (5,),(1,1) )
data = genfromtxt("mountain_car_batch_data.mat")
data[:,5] = squeeze(vRE_reward(data[:,:5]))
policy_RE,omega_RE = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=mountain_car_phi, phi_dim=150, iterations_max=20 )#None,zeros((75,1))#
savetxt("data/Boubou_omega_"+str(NB_SAMPLES)+"_"+RAND_STRING+".mat",omega_RE)
print "Relative entropy : "
print mountain_car_mean_performance(policy_RE)

