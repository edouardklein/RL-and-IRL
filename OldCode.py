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
        
def least_squares_regressor(data, psi, x_dim, _lambda=0.1):
    """Return a function that given a x, will output a y based on a least square regression of the data provided in argument"""
    X = data[:,0:x_dim]
    Y = data[:,x_dim:]
    psi_X = squeeze(psi(X))
    #print psi_X
    omega = dot(dot(inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])),psi_X.transpose()),Y)
    #omega = dot(inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])),psi_X.transpose())
    #omega = inv(dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1]))
    #omega = dot(psi_X.transpose(),psi_X)+_lambda*identity(psi_X.shape[1])
    #print psi_X.shape
    #omega = dot(psi_X.transpose(),psi_X)
    regression = lambda x:dot(omega.transpose(),psi(x))
    return regression
                
def CSI(data, classifier, regressor, A, s_dim=1, gamma=0.9):#FIXME7: does not work for vectorial actions
    column_shape = (len(data),1)
    s = data[:,0:s_dim]
    a = data[:,s_dim:s_dim+1].reshape(column_shape)
    sa = data[:,0:s_dim+1]
    s_dash = data[:,s_dim+1:s_dim+1+s_dim]
    pi_c,q = classifier(hstack([s,a]))
    a_dash = pi_c(s_dash).reshape(column_shape)
    sa_dash = hstack([s_dash,a_dash])
    hat_r = (q(sa)-gamma*q(sa_dash)).reshape(column_shape)
    print "Shapes : s, a, s_dash, a_dash, sa, sa_dash, hat_r"+str([x.shape for x in [s,a,s_dash,a_dash,sa, sa_dash,hat_r]])
    r_min = min(hat_r)-ones(column_shape)
    regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in A] 
    def add_output_column( reg_mat ):
        actions = reg_mat[:,-1].reshape(column_shape)
        hat_r_bool_table = array(actions==a)
        r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
        output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
        return hstack([reg_mat,output_column])
    regression_matrix = vstack(map(add_output_column,regression_input_matrices))
    return regressor( regression_matrix )

    def get_structured_classifier_for_SCI(s_dim, phi, phi_dim, A):
    """Return a classifier function as expected by the CSI algorithm using the StructuredClassifier class"""
    def classifier(data):
        structured_classifier = StructuredClassifier(data, s_dim, phi, phi_dim, A)
        classif_rule,omega_classif = structured_classifier.run()
        score = lambda sa: squeeze(dot(omega_classif.transpose(),phi(sa)))
        return classif_rule,score
    return classifier

def get_least_squares_regressor_for_SCI(phi, sa_dim):
    #ax=b,find x
    def regressor(data):
        a=squeeze(phi(data[:,0:sa_dim]))
        b=data[:,sa_dim:]
        print "Regressor : data, a and b shapes and s_dim: "+str(data.shape)+str(a.shape)+str(b.shape)+str(sa_dim)
        x=lstsq(a,b)[0]
        print "Regressor : x.shape : "+str(x.shape)
        return lambda sa:dot(x.transpose(),phi(sa))
    return lambda data: regressor(data)


#VRAC
states = rand(1000,2)*6-3
actions = policy(states)
data = hstack([states,actions])
classifier = get_structured_classifier_for_SCI(2, inverted_pendulum_phi, 30, ACTION_SPACE)
classif_rule,score = classifier(data)
data_0 = array([l for l in data if l[2]==0.])
data_1 = array([l for l in data if l[2]==1.])
data_2 = array([l for l in data if l[2]==2.])
inverted_pendulum_plot_policy(policy)
plot(data_0[:,0],data_0[:,1],ls='',marker='o')
plot(data_1[:,0],data_1[:,1],ls='',marker='o')
plot(data_2[:,0],data_2[:,1],ls='',marker='o')
inverted_pendulum_plot_policy( classif_rule )
two_args_score = lambda p,s: score(zip_stack(p,s,squeeze(classif_rule(zip_stack(p,s)))))
inverted_pendulum_plot(two_args_score)


reg_S = rand(1000,2)*2-1
reg_A = policy(reg_S)
reg_X = hstack([reg_S,reg_A])
test_omega = rand(30,1)-0.5
reg_Y = dot(test_omega.transpose(),inverted_pendulum_phi(reg_X)).reshape(1000,1)

regressor = get_least_squares_regressor_for_SCI(inverted_pendulum_phi, 3)
reg = regressor(hstack([reg_X,reg_Y]))

inverted_pendulum_plot_SAReward( reg, policy)

X = linspace(-pi,pi,30)
Y = X
X,Y = meshgrid(X,Y)
XY=zip_stack(X,Y)
A = squeeze(policy(XY))
B = zip_stack(X,Y,A)
Z = squeeze(reg(B))
from matplotlib import cm
fig = figure()
ax=Axes3D(fig)
ax.plot_surface(X,Y,Z, cmap=cm.jet,cstride=1,rstride=1)
ax.scatter(reg_X[:,0],reg_X[:,1],reg_Y,c=reg_Y)
show()


#Déroulage "à la main" de l'algo CSI, pour comprendre ce qui cloche
#def CSI(data, classifier, regressor, A, s_dim=1, gamma=0.9):#FIXME7: does not work for vectorial actions
#data = data_expert[:50,:]

data = rand(2000,2)*2*pi - pi
data_classif = hstack([data,policy(data)])
data_classif = hstack([data_classif,zeros((2000,2))])
for index, line in enumerate(data_classif):
    data_classif[index,-2:] = inverted_pendulum_next_state(data_classif[index,:2],data_classif[index,2])
random_policy= lambda s:choice(ACTION_SPACE)
vrandom_policy = non_scalar_vectorize( random_policy, (2,), (1,1) )
pi_r = lambda state: vrandom_policy(state).reshape(state.shape[:-1]+(1,))
data_regress = hstack([data,pi_r(data)])
data_regress = hstack([data_regress,zeros((2000,2))])
for index, line in enumerate(data_regress):
    data_regress[index,-2:] = inverted_pendulum_next_state(data_regress[index,:2],data_regress[index,2])

gamma=0.9
#classifier, regressor, deja definis
A = ACTION_SPACE
s_dim=2
phi=inverted_pendulum_phi
psi=inverted_pendulum_psi


#Algo
column_shape = (len(data_classif),1)
s = data_classif[:,0:s_dim]
a = data_classif[:,s_dim:s_dim+1].reshape(column_shape)
sa = data_classif[:,0:s_dim+1]
s_dash = data_classif[:,s_dim+1:s_dim+1+s_dim]
#from sklearn import svm
#clf = svm.SVC(C=1000., probability=True)
#clf.fit(squeeze(psi(s)), a)
#clf_predict= lambda state : clf.predict(squeeze(psi(state)))
#vpredict = non_scalar_vectorize( clf_predict, (2,), (1,1) )
#pi_c = lambda state: vpredict(state).reshape(state.shape[:-1]+(1,))
#clf_score = lambda sa : squeeze(clf.predict_proba(squeeze(psi(sa[:2]))))[sa[-1]]
#vscore = non_scalar_vectorize( clf_score,(3,),(1,1) )
#q = lambda sa: vscore(sa).reshape(sa.shape[:-1])
pi_c,q=policy,lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
#pi_c,q = classifier(hstack([s,a]))
#Plots de la politique de l'expert, des données fournies par l'expert, de la politique du classifieur
inverted_pendulum_plot_policy(policy)
scatter(data_classif[:,0],data_classif[:,1],c=data_classif[:,2])
figure()
scatter(data_classif[:,0],data_classif[:,1],c=data_classif[:,2])
inverted_pendulum_plot_policy(pi_c)
##Plots de Q et de la fonction de score du classifieur et évaluation de la politique du classifieur
#phi=inverted_pendulum_phi
Q = lambda sa: squeeze(dot(omega.transpose(),phi(sa)))
Q_0 = lambda p,s:Q(zip_stack(p,s,0*ones(p.shape)))
Q_1 = lambda p,s:Q(zip_stack(p,s,1*ones(p.shape)))
Q_2 = lambda p,s:Q(zip_stack(p,s,2*ones(p.shape)))
q_0 = lambda p,s:q(zip_stack(p,s,0*ones(p.shape)))
q_1 = lambda p,s:q(zip_stack(p,s,1*ones(p.shape)))
q_2 = lambda p,s:q(zip_stack(p,s,2*ones(p.shape)))
inverted_pendulum_plot(Q_0)
inverted_pendulum_plot(Q_1)
inverted_pendulum_plot(Q_2)
inverted_pendulum_plot(q_0)
inverted_pendulum_plot(q_1)
inverted_pendulum_plot(q_2)
##FIXME: combien de fois le classifieur tombe-t-il ?


#On  continue CSI
s = data_regress[:,0:s_dim]
a = data_regress[:,s_dim:s_dim+1].reshape(column_shape)
sa = data_regress[:,0:s_dim+1]
s_dash = data_regress[:,s_dim+1:s_dim+1+s_dim]
a_dash = pi_c(s_dash).reshape(column_shape)
sa_dash = hstack([s_dash,a_dash])
hat_r = (q(sa)-gamma*q(sa_dash)).reshape(column_shape)
r_min = min(hat_r)-1.*ones(column_shape)
#Plot des samples hat_r Pour chacune des 3 actions
sar = hstack([sa,hat_r])
for action in ACTION_SPACE:
    sr = array([l for l in sar if l[2]==action])
    axis([-pi,pi,-pi,pi])
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
    colorbar()
    figure()
#On continue SCI

##Avec l'heuristique : 
#regression_input_matrices = [hstack([s,action*ones(column_shape)]) for action in A] 
#def add_output_column( reg_mat ):
#    actions = reg_mat[:,-1].reshape(column_shape)
#    hat_r_bool_table = array(actions==a)
#    r_min_bool_table = array(hat_r_bool_table==False) #"not hat_r_bool_table" does not work as I expected
#    output_column = hat_r_bool_table*hat_r+r_min_bool_table*r_min
#    return hstack([reg_mat,output_column])
#regression_matrix = vstack(map(add_output_column,regression_input_matrices))
#On plotte les mêmes données que juste précedemment, mais avec l'heuristique en prime
#for action in ACTION_SPACE:
#    sr = array([l for l in regression_matrix if l[2]==action])
#    axis([-pi,pi,-pi,pi])
#    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', cmap = cm.jet );
#    colorbar()
#    figure()

##Sans heuristique :
regression_matrix=sar



#On continue CSI
def regressor(data):
    a=squeeze(phi(data[:,0:3]))
    #a=data[:,0:3]
    b=data[:,3:]
#    print x.shape
    #return lambda sa:dot(x,phi(sa))
    ##Trichons
    def triche(sa):
        s_dash = inverted_pendulum_next_state(sa[:2],sa[-1])
        a_dash = pi_c(s_dash)
        sa_dash=hstack([s_dash,a_dash])
        return q(sa)-gamma*q(sa_dash)
    vtriche=non_scalar_vectorize(triche, (3,), (1,1))
    return lambda sa:vtriche(sa)
reg = regressor( regression_matrix )
CSI_reward = lambda sas: squeeze(reg(sas[:3]))
#On plotte les rewards en fonction de l'action
for action in ACTION_SPACE:
    sr = array([l for l in regression_matrix if l[2]==action])
    R = lambda p,s: squeeze( reg(zip_stack(p,s,action*ones(p.shape))))
    pos = linspace(-pi,pi,30)
    speed = linspace(-pi,pi,30)
    pos,speed = meshgrid(pos,speed)
    Z = R(pos,speed)
    figure()
    contourf(pos,speed,Z,50)
    scatter(sr[:,0],sr[:,1],s=20,c=sr[:,3], marker = 'o', )#cmap = cm.jet );
    clim(vmin=min(Z.reshape(-1)),vmax=max(Z.reshape(-1)))
    colorbar()
def mean_reward(s,p):
    actions = [a*ones(s.shape) for a in ACTION_SPACE]
    matrices = [zip_stack(s,p,a) for a in actions]
    return mean(array([squeeze(reg(m)) for m in matrices]), axis=0)
inverted_pendulum_plot(mean_reward)



def inverted_pendulum_expert_trace( reward ):
    data = inverted_pendulum_random_trace(reward=reward)
    policy,omega = lspi( data, s_dim=2,a_dim=1, A=ACTION_SPACE, phi=inverted_pendulum_phi, phi_dim=30, iterations_max=10 )
    inverted_pendulum_plot(inverted_pendulum_V(omega))
    two_args_pol = lambda p,s:squeeze(policy(zip_stack(p,s)))
    inverted_pendulum_plot(two_args_pol,contour_levels=3)
    return inverted_pendulum_trace( policy,run_length=EXPERT_RUN_LENGTH ),policy,omega
data_expert,policy,omega = inverted_pendulum_expert_trace(inverted_pendulum_reward)
#data_CSI,policy_CSI,omega_CSI = inverted_pendulum_expert_trace(CSI_reward)
CSI_falls_rate = - mean( data_CSI[:,5] )
print "Rate of falls for IRL controller "+str(CSI_falls_rate)



