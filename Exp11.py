# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

X=[10, 30, 100, 300, 700, 1000]
CSI_data = genfromtxt("data/CSI_X_10_30_100_300_700_1000.mat")
SCIRL_data = genfromtxt("data/SCIRL_X_10_30_100_300_700_1000.mat")
SCIRLMC_data = genfromtxt("data/SCIRLMC_X_10_30_100_300_700_1000.mat")
RE_data = genfromtxt("data/RE_X_10_30_100_300_700_1000.mat")
Classif_data = genfromtxt("data/Classif_X_10_30_100_300_700_1000.mat")
Expert_data = genfromtxt("data/Expert_X_10_30_100_300_700_1000.mat")

# <codecell>

plot(X,mean(CSI_data,axis=1),color='red')
plot(X,mean(SCIRL_data,axis=1),color='orange')
plot(X,mean(SCIRLMC_data,axis=1),color='orange',ls='--')
plot(X,mean(Classif_data,axis=1),color='green')
plot(X,mean(RE_data,axis=1),color='blue')
plot(X,mean(Expert_data,axis=1),color='pink')

# <codecell>

plot(X,CSI_data.min(axis=1),color='red')
plot(X,SCIRL_data.min(axis=1),color='orange')
plot(X,SCIRLMC_data.min(axis=1),color='orange',ls='-')
plot(X,Classif_data.min(axis=1),color='green')
plot(X,RE_data.min(axis=1),color='blue')
plot(X,Expert_data.min(axis=1),color='pink')

# <codecell>

plot(X,CSI_data.max(axis=1),color='red')
plot(X,SCIRL_data.max(axis=1),color='orange')
plot(X,SCIRLMC_data.max(axis=1),color='orange',ls='-')
plot(X,Classif_data.max(axis=1),color='green')
plot(X,RE_data.max(axis=1),color='blue')
plot(X,Expert_data.max(axis=1),color='pink')

# <codecell>

#On va partir sur la moyenne seule
rc('text', usetex=True)
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['legend.fontsize'] = 'medium'
plot(X[:4],mean(CSI_data,axis=1)[:4],"ro-",mec="r", mfc="w", lw=5, mew=3, ms=10, label=r"CSI")
plot(X[:4],mean(SCIRLMC_data,axis=1)[:4],"v--",color='orange',mec="orange", mfc="w", lw=3, mew=3, ms=9, label="SCIRL")
plot(X[:4],mean(RE_data,axis=1)[:4],"^-.",color='blue',mec="blue", mfc="w", lw=3, mew=3, ms=9, label=r"Relative Entropy")
plot(X[:4],mean(Classif_data,axis=1)[:4],"s:",color='green',mec="green", mfc="w", lw=3, mew=3, ms=9, label="Classification")
plot(X[:4],mean(Expert_data,axis=1)[:4],"p-",color='pink',mec="pink", mfc="w", lw=3, mew=3, ms=9, label="Expert")
axis([0,310,40,250])
legend()
xlabel(r"Number of samples from the expert")
ylabel("Average length of episode")
grid()
savefig("Exp11.pdf")

# <codecell>

from matrix2latex import matrix2latex
m = [[1, 1], [2, 4], [3, 9]] # python nested list
t = matrix2latex(m)
print t
X

# <codecell>

from scipy.stats import ks_2samp
table = zeros((4,2))
for i in range(0,len(X[:4])):
    y_csi = CSI_data[i]
    y_scirl = SCIRL_data[i]
    pvalue = ks_2samp(y_csi,y_scirl)[1]
    table[i,0] = X[i]
    table[i,1] = pvalue
t = matrix2latex(table, headerRow=["Number of expert samples","$p$-value"])
print t

# <codecell>

#plotting the mountain car data
def filled_mean_min_max(X, Y_mean, Y_min, Y_max, color, _alpha, style, lblmain,lblminmax ):
    "Plot data, with bold mean line, and a light color fill betwee the min and max"
    if lblmain == None:
        plot( X, Y_mean,color=color,lw=2)
    else:
        plot( X, Y_mean,color=color,lw=2, label=lblmain)
    if lblminmax == None:
        plot( X, Y_min, color=color,lw=1,linestyle=style)
    else:
        plot( X, Y_min, color=color,lw=1,linestyle=style, label=lblminmax)
    plot( X, Y_max, color=color,lw=1,linestyle=style)
    fill_between(X,Y_min,Y_max,facecolor=color,alpha=_alpha)

#filled_mean_min_max(X,Y_mean_CSI, Y_mean_CSI-Y_deviation_CSI, Y_mean_CSI+Y_deviation_CSI,'red',
                    #0.4,'-.',None,None)
filled_mean_min_max(X,Y_mean_CSI, Y_min_CSI, Y_max_CSI,'red',
                    0.2,'--',None,None)

#filled_mean_min_max(X,Y_mean_Class, Y_mean_Class-Y_deviation_Class, Y_mean_Class+Y_deviation_Class,'blue',
                    #0.4,'-.',None,None)
filled_mean_min_max(X,Y_mean_Class, Y_min_Class, Y_max_Class,'blue',
                    0.2,'--',None,None)
#plot( X,Y_expert*ones(array(X).shape) , color='purple',lw=5)
filled_mean_min_max(X,Y_mean_SCIRLMC, Y_min_SCIRLMC, Y_max_SCIRLMC,'green',
                    0.2,'--',None,None)
filled_mean_min_max(X,Y_mean_RE, Y_min_RE, Y_max_RE,'orange',
                    0.2,'--',None,None)
figure()
filled_mean_min_max(X,Y_mean_Expert, Y_min_Expert, Y_max_Expert,'pink',
                    0.2,'--',None,None)
figure()

axis([10,1000,0,310])
figure()



figure()
sort(all_data_CSI[0])
datasets = map(sort,all_data_CSI)
datasets_len = map(len,datasets)
lim = min(datasets_len)/2
print datasets_len
for i in range(0,lim):
    #print "trun "+str(i)+" len "+str(len(datasets[0]))
    y_min = [d[0] for d in datasets]
    y_max = [d[-1] for d in datasets]
    fill_between(X,y_min,y_max,facecolor='red',alpha=1./float(lim))
    axis([10,1000,0,310])
    datasets = [d[1:-1] for d in datasets]

figure()
sort(all_data_CSI[0])
datasets = map(sort,all_data_Classif)
datasets_len = map(len,datasets)
lim = min(datasets_len)/2
print datasets_len
for i in range(0,lim):
    #print "trun "+str(i)+" len "+str(len(datasets[0]))
    y_min = [d[0] for d in datasets]
    y_max = [d[-1] for d in datasets]
    fill_between(X,y_min,y_max,facecolor='blue',alpha=1./float(lim))
    axis([10,1000,0,310])
    datasets = [d[1:-1] for d in datasets]
    
figure()
sort(all_data_Expert[0])
datasets = map(sort,all_data_Expert)
datasets_len = map(len,datasets)
lim = min(datasets_len)/2
print datasets_len
for i in range(0,lim):
    #print "trun "+str(i)+" len "+str(len(datasets[0]))
    y_min = [d[0] for d in datasets]
    y_max = [d[-1] for d in datasets]
    fill_between(X,y_min,y_max,facecolor='pink',alpha=1./float(lim))
    axis([10,1000,0,310])
    datasets = [d[1:-1] for d in datasets]
    

