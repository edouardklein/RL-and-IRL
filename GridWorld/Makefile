
GridWorld.h: GridWorld.org
	$(call tangle,"GridWorld.org")

GridWorld_clean:
	find . -maxdepth 1 -iname "GridWorld.h"   | xargs $(XARGS_OPT) rm

../DP_mu.py:
	make -C .. DP_mu.py

../DP.py:
	make -C .. DP.py

../LAFEM.py:
	make -C .. LAFEM.py

LAFEM_Exp1_run: LAFEM_Exp1.py ../DP_mu.py ../DP.py Pi2txt.py ../a2str.py ../LAFEM.py TT_5x5_expertPGen.py
	python LAFEM_Exp1.py

V_expert.pdf: LAFEM_Exp1_V_expert.gp V_expert.txt
	gnuplot LAFEM_Exp1_V_expert.gp
	ps2pdf V_expert.ps
	rm V_expert.ps

V_agent.pdf: LAFEM_Exp1_V_agent.gp V_agent.txt
	gnuplot LAFEM_Exp1_V_agent.gp
	ps2pdf V_agent.ps
	rm V_agent.ps

true_reward.pdf: LAFEM_Exp1_true_reward.gp true_reward.txt
	gnuplot LAFEM_Exp1_true_reward.gp
	ps2pdf true_reward.ps
	rm true_reward.ps

retrieved_reward.pdf: LAFEM_Exp1_retrieved_reward.gp retrieved_reward.txt
	gnuplot LAFEM_Exp1_retrieved_reward.gp
	ps2pdf retrieved_reward.ps
	rm retrieved_reward.ps


V_expert.txt: LAFEM_Exp1_run
V_agent.txt: LAFEM_Exp1_run
true_reward.txt: LAFEM_Exp1_run
retrieved_reward.txt: LAFEM_Exp1_run

ErrorBars: conf.py criteria_lstd.dat both_error.dat 
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

criteria_mc.tex: criteria_mc.pdf
	cat criteria_mc.tex | sed "s|criteria_mc|../Code/GridWorld/criteria_mc|g" > tmp.tex && mv tmp.tex criteria_mc.tex

criteria_mc.pdf: criteria_mc.eps
	epstopdf criteria_mc.eps

criteria_mc.eps: criteria_mc.dat criteria_mc.gp
	gnuplot criteria_mc.gp

criteria_mc.dat: plot.dat
	cat plot.dat | grep "^500" | sed 's/500 //'> criteria_mc.dat

criteria_lstd_EB.tex: criteria_lstd_EB.pdf
	cat criteria_lstd_EB.tex | sed "s|criteria_lstd_EB|../Code/GridWorld/criteria_lstd_EB|g" > tmp.tex && mv tmp.tex criteria_lstd_EB.tex

criteria_lstd_EB.pdf: criteria_lstd_EB.eps
	epstopdf criteria_lstd_EB.eps

criteria_lstd_EB.eps: criteria_lstd.dat-0 criteria_lstd.dat-1 criteria_lstd.dat-2 criteria_lstd.dat-3 criteria_lstd_EB.gp
	gnuplot criteria_lstd_EB.gp

criteria_lstd.dat-0: ErrorBars

criteria_lstd.dat-1: ErrorBars

criteria_lstd.dat-2: ErrorBars

criteria_lstd.dat-3: ErrorBars

criteria_lstd.tex: criteria_lstd.pdf
	cat criteria_lstd.tex | sed "s|criteria_lstd|../Code/GridWorld/criteria_lstd|g" > tmp.tex && mv tmp.tex criteria_lstd.tex

criteria_lstd.pdf: criteria_lstd.eps
	epstopdf criteria_lstd.eps

criteria_lstd.eps: criteria_lstd.dat criteria_lstd.gp
	gnuplot criteria_lstd.gp

criteria_lstd.dat: plot.dat
	cat plot.dat | grep -E "^B " | sed "s/B //"> criteria_lstd.dat

both_error_EB.tex: both_error_EB.pdf
	cat both_error_EB.tex | sed "s|both_error_EB|../Code/GridWorld/both_error_EB|g" > tmp.tex && mv tmp.tex both_error_EB.tex

both_error_EB.pdf: both_error_EB.eps
	epstopdf both_error_EB.eps

both_error_EB.eps: both_error.dat-0 both_error.dat-1 both_error_EB.gp
	gnuplot both_error_EB.gp

both_error.dat-0: ErrorBars

both_errot.dat-1: ErrorBars

both_error.tex: both_error.pdf
	cat both_error.tex | sed "s|both_error|../Code/GridWorld/both_error|g" > tmp.tex && mv tmp.tex both_error.tex

both_error.pdf: both_error.eps
	epstopdf both_error.eps

both_error.eps: both_error.dat both_error.gp
	gnuplot both_error.gp

both_error.dat: criteria_mc_samples.dat criteria_lstd.dat
	cat criteria_lstd.dat | cut -d" " -f'1 4' > tmp &&\
	cat criteria_mc_samples.dat | cut -d" " -f'1 4' > tmp2 &&\
	paste -d' ' tmp tmp2 > both_error.dat

criteria_mc_samples.dat: plot.dat
	cat plot.dat | grep -E "^C " | sed "s/C //"> criteria_mc_samples.dat

TT_5x5_PENorth.mat: TT_5x5_expertPGen.py ../a2str.py
	python TT_5x5_expertPGen.py 
TT_5x5_PESouth.mat: TT_5x5_expertPGen.py ../a2str.py
	python TT_5x5_expertPGen.py 
TT_5x5_PEEast.mat: TT_5x5_expertPGen.py ../a2str.py
	python TT_5x5_expertPGen.py 
TT_5x5_PEWest.mat: TT_5x5_expertPGen.py ../a2str.py
	python TT_5x5_expertPGen.py 
TT_5x5_Ppi.mat: TT_5x5_expertPGen.py ../a2str.py
	python TT_5x5_expertPGen.py 

TT_5x5_Rewards.mat: TT_Exp1_run

TT_5x5_R1.mat: TT_Exp1_run
TT_5x5_R2.mat: TT_Exp1_run
TT_5x5_R3.mat: TT_Exp1_run


TT_Exp1_run: ../Constraint.py TT_5x5_Ppi.mat TT_5x5_PENorth.mat TT_5x5_PEWest.mat TT_5x5_PESouth.mat TT_5x5_PEEast.mat
	python ../Constraint.py TT_5x5_Ppi.mat TT_5x5_PENorth.mat TT_5x5_PEWest.mat TT_5x5_PESouth.mat TT_5x5_PEEast.mat > TT_5x5_C.mat

	python ../TaskTransfer_SF.py TT_5x5_C.mat > TT_5x5_Rewards.mat
	rm TT_5x5_C.mat

	python TT_5x5_mat2gp.py TT_5x5_Rewards.mat 0 > TT_5x5_R1.mat
	python TT_5x5_mat2gp.py TT_5x5_Rewards.mat 1 > TT_5x5_R2.mat
	python TT_5x5_mat2gp.py TT_5x5_Rewards.mat 2 > TT_5x5_R3.mat

TT_5x5_R1.pdf: TT_5x5_R_pdf
TT_5x5_R2.pdf: TT_5x5_R_pdf
TT_5x5_R3.pdf: TT_5x5_R_pdf

TT_5x5_R_pdf: TT_5x5_R1.mat TT_5x5_R2.mat TT_5x5_R3.mat
	gnuplot TT_5x5_R1.gp
	ps2pdf TT_5x5_R1.ps
	rm TT_5x5_R1.ps
	gnuplot TT_5x5_R2.gp
	ps2pdf TT_5x5_R2.ps
	rm TT_5x5_R2.ps
	gnuplot TT_5x5_R3.gp
	ps2pdf TT_5x5_R3.ps
	rm TT_5x5_R3.ps

TT_Exp1: TT_Exp1_run TT_5x5_R1.pdf TT_5x5_R2.pdf TT_5x5_R3.pdf
	python TT_5x5_agentAsExpert.py

../Constraint.py:
	make -C .. Constraint.py
../a2str.py:
	make -C .. a2str.py

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs

LAFEM_Exp1.py: LAFEM_Exp1.org 
	$(call tangle,"LAFEM_Exp1.org")

Pi2txt.py: LAFEM_Exp1.org
	$(call tangle,"LAFEM_Exp1.org")

LAFEM_Exp1_V_expert.gp: LAFEM_Exp1.org
	$(call tangle,"LAFEM_Exp1.org")

plot.c: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
criteria_mc.gp: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
criteria_lstd.gp: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
criteria_lstd_EB.gp: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
both_error_EB.gp: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
both_error.gp: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")
conf.py: LSTDmu_Exp.org 
	$(call tangle,"LSTDmu_Exp.org")

generator.c: generator.org 
	$(call tangle,"generator.org")

simulator.c: simulator.org 
	$(call tangle,"simulator.org")
simulator.h: simulator.org 
	$(call tangle,"simulator.org")

TT_5x5_expertPGen.py: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")
TT_5x5_mat2gp.py: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")
TT_5x5_R1.gp: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")
TT_5x5_R2.gp: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")
TT_5x5_R3.gp: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")
TT_5x5_agentAsExpert.py: TT_Exp1.org 
	$(call tangle,"TT_Exp1.org")

plot.o: plot.c GridWorld.h ../utils.h ../LSPI.h ../greedy.h simulator.h ../abbeel2004apprenticeship.h ../LSTDmu.h ../LSTDQ.h ../criteria.h ../RL_Globals.h ../IRL_Globals.h
	$(call c2obj,"plot.c")

generator.o: generator.c GridWorld.h ../utils.h 
	$(call c2obj,"generator.c")

simulator.o: simulator.c simulator.h GridWorld.h ../utils.h 
	$(call c2obj,"simulator.c")

O2EXE=gcc $(LFLAGS)
plot.exe: plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o
	$(O2EXE) -o plot.exe plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o

generator.exe: generator.o ../utils.o
	$(O2EXE) -o generator.exe generator.o ../utils.o

LAFEM_Exp1_clean:
	find . -maxdepth 1 -iname "LAFEM_Exp1.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Pi2txt.py"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "Pi2txt.pyc"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "LAFEM_Exp1_V_expert.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_expert.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_expert.txt"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_expert.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp1_V_agent.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_agent.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_agent.txt"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_agent.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp1_true_reward.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "true_reward.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "true_reward.txt"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM_Exp1_retrieved_reward.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "retrieved_reward.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "retrieved_reward.txt"   | xargs $(XARGS_OPT) rm

LSTDmu_Exp_clean:
	find . -maxdepth 1 -iname "criteria_mc.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd_EB.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "tmp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "tmp2"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "plot.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "plot.samples"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Samples.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_mc.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_mc.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_mc.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_mc.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "conf.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd_EB.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd_EB.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd_EB.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.dat-0"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.dat-1"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.dat-2"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.dat-3"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_lstd.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.dat-0"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.dat-1"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "both_error_EB.eps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria_mc_samples.dat"   | xargs $(XARGS_OPT) rm


Generator_clean:
	find . -maxdepth 1 -iname "Generator.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Generator.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "Generator.o"   | xargs $(XARGS_OPT) rm

Simulator_clean:
	find . -maxdepth 1 -iname "simulator.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "simulator.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "simulator.o"   | xargs $(XARGS_OPT) rm

TT_Exp1_clean:
	find .. -maxdepth 1 -iname "a2str.pyc"   | xargs $(XARGS_OPT) rm
	find .. -maxdepth 1 -iname "DP.pyc"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_agentAsExpert.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R3.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R2.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R1.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_mat2gp.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PENorth.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PESouth.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PEWest.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PEEast.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_Ppi.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_expertPGen.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R1.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R2.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R3.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R1.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R2.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_R3.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PAEast.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PANE.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PANW.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PANorth.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PASE.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PASW.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PASouth.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PAWest.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PESouth.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PENorth.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PEWest.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_PEEast.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_Rewards.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_5x5_Ppi.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "V_expert.mat"   | xargs $(XARGS_OPT) rm

clean: GridWorld_clean LAFEM_Exp1_clean LSTDmu_Exp_clean Generator_clean Simulator_clean TT_Exp1_clean

nuke: #You probably don't want to make this one
	find ErrorBars/ -maxdepth 1 -iname "*.dat*" | xargs $(XARGS_OPT) rm

Makefile:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org

  CFLAGS+= -I..
c2obj=gcc $(CFLAGS) -c $1
