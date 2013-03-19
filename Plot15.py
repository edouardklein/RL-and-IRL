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
for x in X:
    Y = genfromtxt("data/Exp14_"+str(x)+".mat")
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
legend(bbox_to_anchor=(1, 0.85))
xlabel(r"Number of samples from the expert")
ylabel("Average performance")
grid()
axis([0,410,-2.2,8])
savefig("Exp14.pdf")

