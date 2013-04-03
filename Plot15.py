# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Plotting IRL perf on the Highway
X=array([3, 7, 10, 15, 20])
Abcissas = X*X #N episodes of length N => N*N samples
Y_random = []
Y_Classif = []
Y_RE = []
Y_SCIRL = []
Y_CSI = []
Y_expert = []
for x in X:
    Y = genfromtxt("data/Exp14_"+str(x)+".mat")
    Y_random.append(Y[:,0])
    Y_Classif.append(Y[:,1])
    Y_RE.append(Y[:,2])
    Y_SCIRL.append(Y[:,3])
    Y_CSI.append(Y[:,4])
    Y_expert.append(Y[:5])
#On va partir sur la moyenne seule
rc('text', usetex=True)
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['legend.fontsize'] = 'medium'
plot(Abcissas,map(mean,Y_CSI),"ro-",mec="r", mfc="w", lw=5, mew=3, ms=10, label=r"CSI")
plot(Abcissas,map(mean,Y_SCIRL),"v--",color='orange',mec="orange", mfc="w", lw=3, mew=3, ms=9, label="SCIRL")
plot(Abcissas,map(mean,Y_RE),"^-.",color='blue',mec="blue", mfc="w", lw=3, mew=3, ms=9, label=r"Relative Entropy")
plot(Abcissas,map(mean,Y_Classif),"s:",color='green',mec="green", mfc="w", lw=3, mew=3, ms=9, label="Classification")
plot(Abcissas,map(mean,Y_random),"D-",color='gray',mec="gray", mfc="w", lw=3, mew=3, ms=9, label="Random")
#axis([0,310,40,250])
legend(bbox_to_anchor=(1, 0.85))
xlabel(r"Number of samples from the expert")
ylabel("Average performance")
grid()
axis([0,410,-2.2,8])
savefig("Exp14.pdf")

# <codecell>

from scipy.stats import ks_2samp
from matrix2latex import matrix2latex
table = zeros((len(Abcissas),2))
for i in range(0,len(Abcissas)):
    y_csi = Y_CSI[i]
    y_scirl = Y_SCIRL[i]
    table[i][0] = Abcissas[i]
    table[i][1] = ks_2samp(y_csi,y_scirl)[1]
t = matrix2latex(table, headerRow=["Number of expert samples","$p$-value"])
print t

# <codecell>

def student_t(X1,X2):
    return abs((mean(X1)-mean(X2)))/sqrt(abs(pow(X1.var(),2)/len(X1)-pow(X2.var(),2)/len(X2)))
[student_t(Y_SCIRL[i],Y_CSI[i]) for i in range(0,5)]

# <codecell>

plot(Abcissas[-4:],map(mean,Y_CSI)[-4:],"ro-",mec="r", mfc="w", lw=5, mew=3, ms=10, label=r"CSI")
plot(Abcissas[-4:],map(mean,Y_SCIRL)[-4:],"v--",color='orange',mec="orange", mfc="w", lw=3, mew=3, ms=9, label="SCIRL")
plot(Abcissas[-4:],map(mean,Y_RE)[-4:],"^-.",color='blue',mec="blue", mfc="w", lw=3, mew=3, ms=9, label=r"Relative Entropy")
#axis([0,310,40,250])
legend(loc='lower right')
xlabel(r"Number of samples from the expert")
ylabel("Average performance")
grid()
axis([40,410,6,7.7])
savefig("Exp14_zoom.pdf")

# <codecell>

#Plotting IRL perf on the perturbed Highway
X=array([3, 7, 10, 15, 20, 25, 30])
Abcissas = X*X #N episodes of length N => N*N samples
Y_random = []
Y_Classif = []
Y_RE = []
Y_SCIRL = []
Y_CSI = []
for x in X:
    Y = genfromtxt("data/Exp17_"+str(x)+".mat")
    Y_random.append(Y[:,0])
    Y_Classif.append(Y[:,1])
    Y_RE.append(Y[:,2])
    Y_SCIRL.append(Y[:,3])
    Y_CSI.append(Y[:,4])
#On va partir sur la moyenne seule
rc('text', usetex=True)
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['legend.fontsize'] = 'medium'
plot(Abcissas,map(mean,Y_CSI),"ro-",mec="r", mfc="w", lw=5, mew=3, ms=10, label=r"CSI")
plot(Abcissas,map(mean,Y_SCIRL),"v--",color='orange',mec="orange", mfc="w", lw=3, mew=3, ms=9, label="SCIRL")
plot(Abcissas,map(mean,Y_RE),"^-.",color='blue',mec="blue", mfc="w", lw=3, mew=3, ms=9, label=r"Relative Entropy")
plot(Abcissas,map(mean,Y_Classif),"s:",color='green',mec="green", mfc="w", lw=3, mew=3, ms=9, label="Classification")
plot(Abcissas,map(mean,Y_random),"D-",color='gray',mec="gray", mfc="w", lw=3, mew=3, ms=9, label="Random")
#axis([0,310,40,250])
legend(bbox_to_anchor=(1, 0.55))
xlabel(r"Number of samples from the expert")
ylabel("Average performance")
grid()
axis([0,910,-2.2,4])
savefig("Exp17.pdf")

# <codecell>

def student_t(X1,X2):
    return abs((mean(X1)-mean(X2)))/sqrt(abs(pow(X1.var(),2)/len(X1)-pow(X2.var(),2)/len(X2)))
[student_t(Y_SCIRL[i],Y_CSI[i]) for i in range(0,7)]

