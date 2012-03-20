
ErrorBars: conf.py threshold_lstd.dat threshold_mc.dat
	../../Utils/ErrorBars.py
../utils.o:
	make -C .. utils.o

../greedy.o:
	make -C .. greedy.o

../LSTDQ.o:
	make -C .. LSTDQ.o

../abbeel2004apprenticeship.o:
	make -C .. abbeel2004apprenticeship.o

../LSTDmu.o:
	make -C .. LSTDmu.o

../criteria.o:
	make -C .. criteria.o

../LSPI.o:
	make -C .. LSPI.o

../utils.h:
	make -C .. utils.h

../greedy.h:
	make -C .. greedy.h

../LSTDQ.h:
	make -C .. LSTDQ.h

../abbeel2004apprenticeship.h:
	make -C .. abbeel2004apprenticeship.h

../LSTDmu.h:
	make -C .. LSTDmu.h

../criteria.h:
	make -C .. criteria.h

../LSPI.h:
	make -C .. LSPI.h

../RL_Globals.h:
	make -C .. RL_Globals.h

../IRL_Globals.h:
	make -C .. IRL_Globals.h

plot.dat: plot.samples plot.exe
	./plot.exe > plot.dat

plot.samples: generator.exe 
	./generator.exe > Samples.dat && touch plot.samples

threshold.tex: threshold.pdf
	cat threshold.tex | sed "s|threshold|../Code/InvertedPendulum/threshold|g" > tmp.tex && mv tmp.tex threshold.tex

threshold.pdf: threshold.eps
	epstopdf threshold.eps

threshold.eps: threshold_lstd.dat threshold_mc.dat threshold.gp
	gnuplot threshold.gp

threshold_lstd.dat: plot.dat
	cat plot.dat | grep "^LSTD" | sed 's/LSTD //' | cut -d' ' -f'2 3' > threshold_lstd.dat

threshold_mc.dat: plot.dat
	cat plot.dat | grep "MC" | sed 's/MC //' | cut -d' ' -f'2 3' > threshold_mc.dat


threshold_EB.tex: threshold_EB.pdf
	cat threshold_EB.tex | sed "s|threshold_EB|../Code/InvertedPendulum/threshold_EB|g" > tmp.tex && mv tmp.tex threshold_EB.tex

threshold_EB.pdf: threshold_EB.eps
	epstopdf threshold_EB.eps

threshold_EB.eps: threshold_lstd.dat-0 threshold_mc.dat-0 threshold_EB.gp ErrorBars
	gnuplot threshold_EB.gp

threshold_lstd.dat-0: ErrorBars

threshold_mc.dat-0: ErrorBars


../LAFEM.py:
	make -C .. LAFEM.py

../a2str.py:
	make -C .. a2str.py

LAFEM_Exp2_Samples.dat: generator.exe 
	./generator.exe > LAFEM_Exp2_Samples.dat

LAFEM_Exp2_Expert.exe: LAFEM_Exp2_Expert.o phipsi.o ../utils.o ../greedy.o ../LSTDQ.o ../LSPI.o InvertedPendulum.o simulator.o
	$(O2EXE) -o LAFEM_Exp2_Expert.exe LAFEM_Exp2_Expert.o phipsi.o ../utils.o ../greedy.o ../LSTDQ.o InvertedPendulum.o ../LSPI.o simulator.o

LAFEM_Exp2_Expert.o: LAFEM_Exp2_Expert.c InvertedPendulum.h simulator.h ../utils.h ../LSPI.h ../greedy.h ../RL_Globals.h phipsi.h
	$(call c2obj,"LAFEM_Exp2_Expert.c")

LAFEM_Exp2_Expert.c: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

D_expert.mat: LAFEM_Exp2_Expert.exe LAFEM_Exp2_Samples.dat
	./LAFEM_Exp2_Expert.exe > omega_expert.mat

omega_expert.mat: LAFEM_Exp2_Expert.exe LAFEM_Exp2_Samples.dat
	./LAFEM_Exp2_Expert.exe > omega_expert.mat

LAFEM_Exp2_MuMC.exe: LAFEM_Exp2_MuMC.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o simulator.o
	$(O2EXE) -o LAFEM_Exp2_MuMC.exe LAFEM_Exp2_MuMC.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o  simulator.o

LAFEM_Exp2_MuMC.o: LAFEM_Exp2_MuMC.c InvertedPendulum.h simulator.h ../utils.h ../greedy.h ../RL_Globals.h ../IRL_Globals.h
	$(call c2obj,"LAFEM_Exp2_MuMC.c")

LAFEM_Exp2_MuMC.c: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

mu_E_Left.mat: LAFEM_Exp2_MuMC.exe omega_expert.mat D_expert.mat
	./LAFEM_Exp2_MuMC.exe
mu_E_Right.mat: LAFEM_Exp2_MuMC.exe omega_expert.mat D_expert.mat
	./LAFEM_Exp2_MuMC.exe
mu_E_None.mat: LAFEM_Exp2_MuMC.exe omega_expert.mat D_expert.mat
	./LAFEM_Exp2_MuMC.exe

LAFEM_Exp2.py: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

theta_lafem.mat: LAFEM_Exp2.py ../a2str.py ../LAFEM.py mu_E_Right.mat mu_E_Left.mat mu_E_None.mat D_expert.mat
	python LAFEM_Exp2.py > theta_lafem.mat

omega_lafem.mat: LAFEM_Exp2_Results.exe theta_lafem.mat
	./LAFEM_Exp2_Results.exe theta_lafem.mat > omega_lafem.mat

LAFEM_Exp2_Results.exe: LAFEM_Exp2_Results.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o simulator.o ../LSPI.o ../LSTDQ.o
	$(O2EXE) -o LAFEM_Exp2_Results.exe LAFEM_Exp2_Results.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o  simulator.o ../LSPI.o ../LSTDQ.o

LAFEM_Exp2_Results.o: LAFEM_Exp2_Results.c InvertedPendulum.h simulator.h ../utils.h ../greedy.h ../RL_Globals.h ../IRL_Globals.h ../LSPI.h
	$(call c2obj,"LAFEM_Exp2_Results.c")

LAFEM_Exp2_Results.c: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_true_R.dat: LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat
	./LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat LAFEM_Exp2

LAFEM_Exp2_lafem_R.dat: LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat LAFEM_Exp2

LAFEM_Exp2_Vexpert.dat: LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat
	./LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat LAFEM_Exp2

LAFEM_Exp2_Vagent.dat: LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem.mat omega_lafem.mat LAFEM_Exp2

LAFEM_Exp2_Plot.exe: LAFEM_Exp2_Plot.o phipsi.o ../utils.o ../greedy.o phipsi.o
	$(O2EXE) -o LAFEM_Exp2_Plot.exe LAFEM_Exp2_Plot.o phipsi.o ../utils.o ../greedy.o  

LAFEM_Exp2_Plot.o: LAFEM_Exp2_Plot.c InvertedPendulum.h ../utils.h ../greedy.h ../RL_Globals.h phipsi.h
	$(call c2obj,"LAFEM_Exp2_Plot.c")

LAFEM_Exp2_Plot.c: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_true_R.gp: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_true_R.pdf: LAFEM_Exp2_true_R.gp LAFEM_Exp2_true_R.dat
	gnuplot LAFEM_Exp2_true_R.gp
	ps2pdf LAFEM_Exp2_true_R.ps
	rm LAFEM_Exp2_true_R.ps

LAFEM_Exp2_lafem_R.gp: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_lafem_R.pdf: LAFEM_Exp2_lafem_R.gp LAFEM_Exp2_lafem_R.dat
	gnuplot LAFEM_Exp2_lafem_R.gp
	ps2pdf LAFEM_Exp2_lafem_R.ps
	rm LAFEM_Exp2_lafem_R.ps

LAFEM_Exp2_Vexpert.gp: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_Vexpert.pdf: LAFEM_Exp2_Vexpert.gp LAFEM_Exp2_Vexpert.dat
	gnuplot LAFEM_Exp2_Vexpert.gp
	ps2pdf LAFEM_Exp2_Vexpert.ps
	rm LAFEM_Exp2_Vexpert.ps

LAFEM_Exp2_Vagent.gp: LAFEM_Exp2.org
	$(call tangle,"LAFEM_Exp2.org")

LAFEM_Exp2_Vagent.pdf: LAFEM_Exp2_Vagent.gp LAFEM_Exp2_Vagent.dat
	gnuplot LAFEM_Exp2_Vagent.gp
	ps2pdf LAFEM_Exp2_Vagent.ps
	rm LAFEM_Exp2_Vagent.ps


LAFEM_Exp3_LSTDmu.exe: LAFEM_Exp3_LSTDmu.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o ../LSTDmu.o
	$(O2EXE) -o LAFEM_Exp3_LSTDmu.exe LAFEM_Exp3_LSTDmu.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o  ../LSTDmu.o

LAFEM_Exp3_LSTDmu.o: LAFEM_Exp3_LSTDmu.c InvertedPendulum.h ../LSTDmu.h ../utils.h ../greedy.h ../RL_Globals.h ../IRL_Globals.h phipsi.h
	$(call c2obj,"LAFEM_Exp3_LSTDmu.c")

LAFEM_Exp3_LSTDmu.c: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

omega_mu_E.mat: LAFEM_Exp3_LSTDmu.exe omega_expert.mat D_expert.mat
	./LAFEM_Exp3_LSTDmu.exe > omega_mu_E.mat

LAFEM_Exp3.py: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

theta_lafem_mu.mat: LAFEM_Exp3.py ../a2str.py ../LAFEM.py omega_mu_E.mat D_expert.mat
	python LAFEM_Exp3.py > theta_lafem_mu.mat

omega_lafem_mu.mat: LAFEM_Exp2_Results.exe theta_lafem_mu.mat
	./LAFEM_Exp2_Results.exe theta_lafem_mu.mat > omega_lafem_mu.mat

LAFEM_Exp3_true_R.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp3

LAFEM_Exp3_lafem_R.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp3

LAFEM_Exp3_Vexpert.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp3

LAFEM_Exp3_Vagent.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp3



LAFEM_Exp3_true_R.gp: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

LAFEM_Exp3_true_R.pdf: LAFEM_Exp3_true_R.gp LAFEM_Exp3_true_R.mat
	gnuplot LAFEM_Exp3_true_R.gp
	ps2pdf LAFEM_Exp3_true_R.ps
	rm LAFEM_Exp3_true_R.ps

LAFEM_Exp3_lafem_R.gp: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

LAFEM_Exp3_lafem_R.pdf: LAFEM_Exp3_lafem_R.gp LAFEM_Exp3_lafem_R.mat
	gnuplot LAFEM_Exp3_lafem_R.gp
	ps2pdf LAFEM_Exp3_lafem_R.ps
	rm LAFEM_Exp3_lafem_R.ps

LAFEM_Exp3_Vexpert.gp: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

LAFEM_Exp3_Vexpert.pdf: LAFEM_Exp3_Vexpert.gp LAFEM_Exp3_Vexpert.mat
	gnuplot LAFEM_Exp3_Vexpert.gp
	ps2pdf LAFEM_Exp3_Vexpert.ps
	rm LAFEM_Exp3_Vexpert.ps

LAFEM_Exp3_Vagent.gp: LAFEM_Exp3.org
	$(call tangle,"LAFEM_Exp3.org")

LAFEM_Exp3_Vagent.pdf: LAFEM_Exp3_Vagent.gp LAFEM_Exp3_Vagent.mat
	gnuplot LAFEM_Exp3_Vagent.gp
	ps2pdf LAFEM_Exp3_Vagent.ps
	rm LAFEM_Exp3_Vagent.ps


#N_VALUES=300 600 900 1500 3000
N_VALUES=2 20 100 200 1000 3000
D_expert_X.mat: D_expert.mat #Dummy target
	for i in $(N_VALUES); do head -n $$i D_expert.mat > D_expert_$$i.mat; done
	touch D_expert_X.mat

omega_mu_E_X.mat: D_expert_X.mat LAFEM_Exp3_LSTDmu.exe omega_expert.mat#Dummy target
	for i in $(N_VALUES); do\
		cp D_expert_$$i.mat D_expert.mat &&\
		./LAFEM_Exp3_LSTDmu.exe > omega_mu_E_$$i.mat;\
	done;
	touch omega_mu_E_X.mat

theta_lafem_X.mat: D_expert_X.mat omega_mu_E_X.mat ../LAFEM.py LAFEM_Exp3.py ../a2str.py #Dummy target
	for i in $(N_VALUES); do\
		cp D_expert_$$i.mat D_expert.mat &&\
		cp omega_mu_E_$$i.mat omega_mu_E.mat &&\
		python LAFEM_Exp3.py > theta_lafem_$$i.mat;\
	done;
	touch omega_mu_E_X.mat
LAFEM_Exp4_quality.dat: theta_lafem_X.mat LAFEM_Exp2_Samples.dat LAFEM_Exp2_Results.exe
	touch LAFEM_Exp4_quality.dat
	rm LAFEM_Exp4_quality.dat #The use of >> makes it necessary to RAZ the file
	for i in $(N_VALUES); do\
		(echo -n "$$i " && ./LAFEM_Exp2_Results.exe theta_lafem_$$i.mat >/dev/null)>> LAFEM_Exp4_quality.dat 2>> LAFEM_Exp4_quality.dat;\
	done;
	cat LAFEM_Exp4_quality.dat | grep agent | tr "[:punct:]" " " | sed 's/ Longueur d un run de l agent  //' > toto
	mv toto LAFEM_Exp4_quality.dat
LAFEM_Exp4_quality.gp: LAFEM_Exp4.org
	$(call tangle,"LAFEM_Exp4.org")

LAFEM_Exp4_quality.pdf: LAFEM_Exp4_quality.gp LAFEM_Exp4_quality.dat
	gnuplot LAFEM_Exp4_quality.gp
	ps2pdf LAFEM_Exp4_quality.ps
	rm LAFEM_Exp4_quality.ps
LAFEM_Exp4_conf.py: LAFEM_Exp4.org 
	$(call tangle,"LAFEM_Exp4.org")
LAFEM_Exp4_quality_EB.gp: LAFEM_Exp4.org
	$(call tangle,"LAFEM_Exp4.org")

LAFEM_Exp4_quality_EB.pdf: LAFEM_Exp4_quality_EB.gp LAFEM_Exp4_ErrorBars
	gnuplot LAFEM_Exp4_quality_EB.gp
	ps2pdf LAFEM_Exp4_quality_EB.ps
	rm LAFEM_Exp4_quality_EB.ps
LAFEM_Exp4_ErrorBars: LAFEM_Exp4_conf.py LAFEM_Exp4_quality.dat
	mv LAFEM_Exp4_conf.py conf.py
	../../Utils/ErrorBars.py
	rm conf.py


LAFEM_Exp5.py: LAFEM_Exp5.org
	$(call tangle,"LAFEM_Exp5.org")

theta_lafem_mu.mat: LAFEM_Exp5.py ../a2str.py ../LAFEM.py omega_mu_E.mat D_expert.mat
	python LAFEM_Exp5.py > theta_lafem_mu.mat

omega_lafem_mu.mat: LAFEM_Exp2_Results.exe theta_lafem_mu.mat
	./LAFEM_Exp2_Results.exe theta_lafem_mu.mat > omega_lafem_mu.mat

LAFEM_Exp5_true_R.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp5

LAFEM_Exp5_lafem_R.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp5

LAFEM_Exp5_Vexpert.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp5

LAFEM_Exp5_Vagent.mat: LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat 
	./LAFEM_Exp2_Plot.exe theta_lafem_mu.mat omega_lafem_mu.mat LAFEM_Exp5



LAFEM_Exp5_true_R.gp: LAFEM_Exp5.org
	$(call tangle,"LAFEM_Exp5.org")

LAFEM_Exp5_true_R.pdf: LAFEM_Exp5_true_R.gp LAFEM_Exp5_true_R.mat
	gnuplot LAFEM_Exp5_true_R.gp
	ps2pdf LAFEM_Exp5_true_R.ps
	rm LAFEM_Exp5_true_R.ps

LAFEM_Exp5_lafem_R.gp: LAFEM_Exp5.org
	$(call tangle,"LAFEM_Exp5.org")

LAFEM_Exp5_lafem_R.pdf: LAFEM_Exp5_lafem_R.gp LAFEM_Exp5_lafem_R.mat
	gnuplot LAFEM_Exp5_lafem_R.gp
	ps2pdf LAFEM_Exp5_lafem_R.ps
	rm LAFEM_Exp5_lafem_R.ps

LAFEM_Exp5_Vexpert.gp: LAFEM_Exp5.org
	$(call tangle,"LAFEM_Exp5.org")

LAFEM_Exp5_Vexpert.pdf: LAFEM_Exp5_Vexpert.gp LAFEM_Exp5_Vexpert.mat
	gnuplot LAFEM_Exp5_Vexpert.gp
	ps2pdf LAFEM_Exp5_Vexpert.ps
	rm LAFEM_Exp5_Vexpert.ps

LAFEM_Exp5_Vagent.gp: LAFEM_Exp5.org
	$(call tangle,"LAFEM_Exp5.org")

LAFEM_Exp5_Vagent.pdf: LAFEM_Exp5_Vagent.gp LAFEM_Exp5_Vagent.mat
	gnuplot LAFEM_Exp5_Vagent.gp
	ps2pdf LAFEM_Exp5_Vagent.ps
	rm LAFEM_Exp5_Vagent.ps


LAFEM_Exp7.py: LAFEM_Exp7.org
	$(call tangle,"LAFEM_Exp7.org")

omega_lafem_phi.mat: LAFEM_Exp7.py ../a2str.py ../LAFEM.py D_expert.mat
	python LAFEM_Exp7.py > omega_lafem_phi.mat

LAFEM_Exp7_Results.exe: LAFEM_Exp7_Results.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o simulator.o ../LSTDQ.o
	$(O2EXE) -o LAFEM_Exp7_Results.exe LAFEM_Exp7_Results.o phipsi.o ../utils.o ../greedy.o  InvertedPendulum.o  simulator.o ../LSTDQ.o

LAFEM_Exp7_Results.o: LAFEM_Exp7_Results.c InvertedPendulum.h simulator.h ../utils.h ../greedy.h ../RL_Globals.h ../IRL_Globals.h  phipsi.h
	$(call c2obj,"LAFEM_Exp7_Results.c")

LAFEM_Exp7_Results.c: LAFEM_Exp7.org
	$(call tangle,"LAFEM_Exp7.org")

LAFEM_Exp7_Vphi.dat: LAFEM_Exp7_Results.exe omega_lafem_phi.mat omega_mu_E.mat theta_lafem_mu.mat
	./LAFEM_Exp7_Results.exe omega_lafem_phi.mat theta_lafem_mu.mat omega_mu_E.mat

LAFEM_Exp7_Vmu.dat: LAFEM_Exp7_Results.exe omega_lafem_phi.mat omega_mu_E.mat theta_lafem_mu.mat
	./LAFEM_Exp7_Results.exe omega_lafem_phi.mat theta_lafem_mu.mat omega_mu_E.mat

LAFEM_Exp7_Vphi.gp: LAFEM_Exp7.org
	$(call tangle,"LAFEM_Exp7.org")
LAFEM_Exp7_Vmu.gp: LAFEM_Exp7.org
	$(call tangle,"LAFEM_Exp7.org")

LAFEM_Exp7_Vphi.pdf: LAFEM_Exp7_Vphi.gp LAFEM_Exp7_Vphi.dat
	gnuplot LAFEM_Exp7_Vphi.gp
	ps2pdf LAFEM_Exp7_Vphi.ps
	rm LAFEM_Exp7_Vphi.ps

LAFEM_Exp7_Vmu.pdf: LAFEM_Exp7_Vagent.gp LAFEM_Exp7_Vmu.dat
	gnuplot LAFEM_Exp7_Vmu.gp
	ps2pdf LAFEM_Exp7_Vmu.ps
	rm LAFEM_Exp7_Vmu.ps

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs

generator.c: generator.org 
	$(call tangle,"generator.org")

InvertedPendulum.c: InvertedPendulum.org 
	$(call tangle,"InvertedPendulum.org")
InvertedPendulum.h: InvertedPendulum.org 
	$(call tangle,"InvertedPendulum.org")

simulator.c: simulator.org 
	$(call tangle,"simulator.org")
simulator.h: simulator.org 
	$(call tangle,"simulator.org")

plot.c: plot.org 
	$(call tangle,"plot.org")
threshold.gp: plot.org 
	$(call tangle,"plot.org")
threshold_EB.gp: plot.org 
	$(call tangle,"plot.org")
conf.py: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
phipsi.h: plot.org 
	$(call tangle,"plot.org")
phipsi.c: plot.org 
	$(call tangle,"plot.org")

generator.o: generator.c InvertedPendulum.h ../utils.h 
	$(call c2obj,"generator.c")

InvertedPendulum.o: InvertedPendulum.c InvertedPendulum.h ../utils.h
	$(call c2obj,"InvertedPendulum.c")

simulator.o: simulator.c simulator.h InvertedPendulum.h ../utils.h ../IRL_Globals.h
	$(call c2obj,"simulator.c")

plot.o: plot.c InvertedPendulum.h ../utils.h ../LSPI.h ../greedy.h simulator.h ../abbeel2004apprenticeship.h ../LSTDmu.h ../criteria.h ../RL_Globals.h ../IRL_Globals.h
	$(call c2obj,"plot.c")
phipsi.o: phipsi.c ../RL_Globals.h InvertedPendulum.h
	$(call c2obj,"phipsi.c")

O2EXE=gcc $(LFLAGS)
generator.exe: generator.o ../utils.o InvertedPendulum.o
	$(O2EXE) -o generator.exe generator.o ../utils.o InvertedPendulum.o

plot.exe: plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o InvertedPendulum.o phipsi.o
	$(O2EXE) -o plot.exe plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o InvertedPendulum.o phipsi.o

Generator_clean:
	find . -maxdepth 1 -iname "generator.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "generator.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "generator.o"   | xargs $(XARGS_OPT) rm

InvertedPendulum_clean:
	find . -maxdepth 1 -iname "InvertedPendulum.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "InvertedPendulum.h"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "InvertedPendulum.c"   | xargs $(XARGS_OPT) rm

Simulator_clean:
	find . -maxdepth 1 -iname "simulator.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "simulator.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "simulator.o"   | xargs $(XARGS_OPT) rm

LSTDmu_Exp_clean:
	find . -maxdepth 1 -iname "threshold.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_EB.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.h"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "phipsi.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "plot.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "plot.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.samples"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Samples.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_lstd.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_mc.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_EB.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_EB.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_EB.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_lstd.dat-0"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "threshold_mc.dat-0"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "conf.py"   | xargs $(XARGS_OPT) rm


LAFEM_Exp2_clean:
	find . -maxdepth 1 -iname "LAFEM_Exp2_*"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp2.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_lafem.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_expert.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "D_expert.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "mu_E_Left.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "mu_E_Right.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "mu_E_None.mat"   | xargs $(XARGS_OPT) rm

LAFEM_Exp3_clean:
	find . -maxdepth 1 -iname "LAFEM_Exp3_*"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp3.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_lafem_mu.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_mu.mat"   | xargs $(XARGS_OPT) rm

clean_LAFEM_Exp4_quality.dat: 
	find . -maxdepth 1 -iname "D_expert.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "D_expert_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp4_quality.*"   | xargs $(XARGS_OPT) rm

LAFEM_Exp4_clean: LAFEM_Exp3_clean LAFEM_Exp2_clean
	find . -maxdepth 1 -iname "D_expert_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp4_quality.*"   | xargs $(XARGS_OPT) rm

LAFEM_Exp5_clean:
	find . -maxdepth 1 -iname "LAFEM_Exp5_*"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp5.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_lafem_mu.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_mu.mat"   | xargs $(XARGS_OPT) rm

clean: Generator_clean InvertedPendulum_clean Simulator_clean LSTDmu_Exp_clean LAFEM_Exp2_clean LAFEM_Exp3_clean LAFEM_Exp4_clean LAFEM_Exp5_clean

Makefile:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org

ORG_CODE_FILES=Makefile.org first.org generator.org InvertedPendulum.org plot.org simulator.org

HTML_FILES=$(ORG_CODE_FILES:.org=.html)

doc: $(HTML_FILES)

%.html:%.org
	emacs -batch --visit $*.org --funcall org-export-as-html-and-open --script ~/.emacs

code:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org

clean:
	find . -maxdepth 1 -iname "*.o"       | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.pdf"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.exe"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.dat"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*~"        | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.pyc"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.eps"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*.tex"     | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "first.c"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "plot.c"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "generator.c"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "InvertedPendulum.h"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "InvertedPendulum.c"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "simulator.h"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "simulator.c"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "code"   | xargs $(XARGS_OPT) rm &&\
	find . -maxdepth 1 -iname "*-0"       | xargs $(XARGS_OPT) rm

nuke: #You probably don't want to make this one
	find ErrorBars/ -maxdepth 1 -iname "*.dat*" | xargs $(XARGS_OPT) rm

  CFLAGS+= -I..
c2obj=gcc $(CFLAGS) -c $1
