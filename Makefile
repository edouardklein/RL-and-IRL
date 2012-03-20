
IRL_Globals.h: IRL_Globals.org
	$(call tangle,"IRL_Globals.org")

IRL_Globals_clean:
	find . -maxdepth 1 -iname "IRL_Globals.h"   | xargs $(XARGS_OPT) rm

DP_mu.py: DP_mu.org
	$(call tangle,"DP_mu.org")

DP_mu_clean:
	find . -maxdepth 1 -iname "DP_mu.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "DP_mu.pyc"   | xargs $(XARGS_OPT) rm

RL_Globals.h: RL_Globals.org
	$(call tangle,"RL_Globals.org")

RL_Globals_clean:
	find . -maxdepth 1 -iname "RL_Globals.h"   | xargs $(XARGS_OPT) rm

TT_Exp2: TT_Exp2.py DP.py a2str.py TT.py
	python TT_Exp2.py

TT_Exp3: TT_Exp3.py DP.py a2str.py TT.py
	python TT_Exp3.py

TT_test0: TaskTransfer.py test/TT_CT01.mat test/TT_CT02.mat test/TT_expectedOutT0.mat a2str.py
	python TaskTransfer.py test/TT_CT01.mat | sort > test/TT_outT01.mat
	python TaskTransfer.py test/TT_CT02.mat | sort > test/TT_outT02.mat
	../Utils/matrix_diff.py test/TT_expectedOutT0.mat test/TT_outT01.mat
	../Utils/matrix_diff.py test/TT_expectedOutT0.mat test/TT_outT02.mat
	rm test/TT_outT01.mat
	rm test/TT_outT02.mat

TT_test1: Constraint.py test/TT_PPi1.mat test/TT_PPi2.mat test/TT_PEast.mat test/TT_PWest.mat test/TT_PSouth.mat test/TT_PNorth.mat TaskTransfer.py test/TT_expectedOutT1.mat a2str.py
	python Constraint.py test/TT_PPi1.mat test/TT_PEast.mat test/TT_PWest.mat test/TT_PSouth.mat test/TT_PNorth.mat > test/TT_C1.mat
	python Constraint.py test/TT_PPi2.mat test/TT_PEast.mat test/TT_PWest.mat test/TT_PSouth.mat test/TT_PNorth.mat > test/TT_C2.mat

	cat test/TT_C1.mat test/TT_C2.mat | sort | uniq > test/TT_CBoth.mat

	python TaskTransfer.py test/TT_CBoth.mat | sort > test/TT_outT1.mat
	../Utils/matrix_diff.py test/TT_expectedOutT1.mat test/TT_outT1.mat
	rm test/TT_C1.mat test/TT_C2.mat test/TT_CBoth.mat test/TT_outT1.mat

TT_test2: TT_Test2.py test/TT_PPi1.mat a2str.py
	python TT_Test2.py > test/TT_outT2.mat
	../Utils/matrix_diff.py test/TT_PPi1.mat test/TT_outT2.mat
	rm test/TT_outT2.mat

TT_test3:TT_Test3.py test/TT_PPi2.mat a2str.py
	python TT_Test3.py > test/TT_outT3.mat
	../Utils/matrix_diff.py test/TT_PPi2.mat test/TT_outT3.mat
	rm test/TT_outT3.mat

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs

LSPI.c: LSPI.org 
	$(call tangle,"LSPI.org")

LSPI.h: LSPI.org
	$(call tangle,"LSPI.org")

LSTDQ.c: LSTDQ.org 
	$(call tangle,"LSTDQ.org")

LSTDQ.h: LSTDQ.org
	$(call tangle,"LSTDQ.org")

LSTDmu.c: LSTDmu.org 
	$(call tangle,"LSTDmu.org")

LSTDmu.h: LSTDmu.org
	$(call tangle,"LSTDmu.org")

TT_Exp2.py: TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

TT_Exp3.py: TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

utils.c: utils.org 
	$(call tangle,"utils.org")

utils.h: utils.org
	$(call tangle,"utils.org")

a2str.py: utils.org
	$(call tangle,"utils.org")

DP.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

TaskTransfer.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

TaskTransfer_SF.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

Constraint.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

TT.py :TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

test/TT_CT01.mat: TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

test/TT_CT02.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

test/TT_expectedOutT0.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")


test/TT_PPi1.mat: TaskTransfer.org
	$(call tangle,"TaskTransfer.org")
test/TT_PPi2.mat: TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

test/TT_PEast.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")
test/TT_PWest.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")
test/TT_PNorth.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")
test/TT_PSouth.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

test/TT_expectedOutT1.mat:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

TT_Test2.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")
TT_Test3.py:TaskTransfer.org
	$(call tangle,"TaskTransfer.org")

abbeel2004apprenticeship.c: abbeel2004apprenticeship.org 
	$(call tangle,"abbeel2004apprenticeship.org")

abbeel2004apprenticeship.h: abbeel2004apprenticeship.org 
	$(call tangle,"abbeel2004apprenticeship.org")

criteria.c: criteria.org 
	$(call tangle,"criteria.org")

criteria.h: criteria.org
	$(call tangle,"criteria.org")

greedy.c: greedy.org 
	$(call tangle,"greedy.org")

greedy.h: greedy.org
	$(call tangle,"greedy.org")

LAFEM.py: NouveauxAlgos.org
	$(call tangle,"NouveauxAlgos.org")

export CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl`
export LFLAGS=`pkg-config --libs gsl` -lm -g
c2obj=gcc $(CFLAGS) -c $1
o2exe=gcc $(LFLAGS)

LSPI.o: LSPI.c LSPI.h LSTDQ.h utils.h greedy.h RL_Globals.h
	$(call c2obj,"LSPI.c")

LSTDQ.o: LSTDQ.c LSTDQ.h RL_Globals.h
	$(call c2obj,"LSTDQ.c")

LSTDmu.o: LSTDmu.c LSTDmu.h utils.h criteria.h LSPI.h greedy.h RL_Globals.h IRL_Globals.h
	$(call c2obj,"LSTDmu.c")

utils.o: utils.c utils.h
	$(call c2obj,"utils.c")

abbeel2004apprenticeship.o: abbeel2004apprenticeship.c abbeel2004apprenticeship.h LSPI.h utils.h RL_Globals.h IRL_Globals.h criteria.h
	$(call c2obj,"abbeel2004apprenticeship.c")

criteria.o: criteria.c criteria.h LSTDQ.h utils.h RL_Globals.h IRL_Globals.h abbeel2004apprenticeship.h
	$(call c2obj,"criteria.c")

greedy.o: greedy.c greedy.h RL_Globals.h
	$(call c2obj,"greedy.c")

org2pdf=emacs -batch --visit $1.org --funcall org-export-as-latex --script ~/.emacs && pdflatex $1.tex && bibtex $1 && pdflatex $1.tex && pdflatex $1.tex

NouveauxAlgos.pdf: NouveauxAlgos.org
	$(call org2pdf,"NouveauxAlgos")

export XARGS_OPT=-tr
LSPI_clean:
	find . -maxdepth 1 -iname "LSPI.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LSPI.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "LSPI.o"   | xargs $(XARGS_OPT) rm

LSTDQ_clean:
	find . -maxdepth 1 -iname "LSTDQ.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LSTDQ.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "LSTDQ.o"   | xargs $(XARGS_OPT) rm

LSTDmu_clean:
	find . -maxdepth 1 -iname "LSTDmu.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LSTDmu.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "LSTDmu.o"   | xargs $(XARGS_OPT) rm

TT_Exp2_clean:
	find . -maxdepth 1 -iname "TT_Exp2.py"   | xargs $(XARGS_OPT) rm

TT_Exp3_clean:
	find . -maxdepth 1 -iname "TT_Exp3.py"   | xargs $(XARGS_OPT) rm

utils_clean:
	find . -maxdepth 1 -iname "utils.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "utils.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "utils.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "a2str.py"   | xargs $(XARGS_OPT) rm

NA_clean:
	find . -maxdepth 1 -iname "NouveauxAlgos.aux"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "NouveauxAlgos.bbl"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "NouveauxAlgos.blg"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "NouveauxAlgos.tex"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "NouveauxAlgos.pdf"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "NouveauxAlgos.log"| xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "NouveauxAlgos.toc"| xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM.py" | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "LAFEM.pyc" | xargs $(XARGS_OPT) rm

TT_Test0_clean:
	find test -maxdepth 1 -iname "TT_CT01.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_CT02.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_expectedOutT0.mat" | xargs $(XARGS_OPT) rm

TT_Test1_clean:
	find test -maxdepth 1 -iname "TT_PPi1.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPi2.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPEast.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPWest.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPNorth.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPSouth.mat" | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_expectedOutT1.mat" | xargs $(XARGS_OPT) rm

TT_Test23_clean:
	find . -maxdepth 1 -iname "TT_Test2.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT_Test3.py"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPi1.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PPi2.mat"   | xargs $(XARGS_OPT) rm

TT_clean:
	find . -maxdepth 1 -iname "DP.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TaskTransfer.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TaskTransfer_SF.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Constraint.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "TT.pyc"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PEast.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PWest.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PNorth.mat"   | xargs $(XARGS_OPT) rm
	find test -maxdepth 1 -iname "TT_PSouth.mat"   | xargs $(XARGS_OPT) rm

a2a_clean:
	find . -maxdepth 1 -iname "abbeel2004apprenticeship.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "abbeel2004apprenticeship.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "abbeel2004apprenticeship.o"   | xargs $(XARGS_OPT) rm

criteria_clean:
	find . -maxdepth 1 -iname "criteria.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "criteria.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "criteria.o"   | xargs $(XARGS_OPT) rm

greedy_clean:
	find . -maxdepth 1 -iname "greedy.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "greedy.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "greedy.o"   | xargs $(XARGS_OPT) rm


clean: TT_Test0_clean TT_Test1_clean TT_Test23_clean IRL_Globals_clean LSPI_clean LSTDQ_clean LSTDmu_clean DP_mu_clean NA_clean RL_Globals_clean TT_Exp2_clean TT_Exp3_clean utils_clean a2a_clean criteria_clean greedy_clean TT_clean
	$(MAKE) -C ChainWalk clean   
	$(MAKE) -C GridWorld clean
	$(MAKE) -C InvertedPendulum clean
	$(MAKE) -C Highway clean

LAFEM_Exp1:
	make -C GridWorld V_expert.pdf V_agent.pdf true_reward.pdf retrieved_reward.pdf

LAFEM_Exp2:
	make -C InvertedPendulum LAFEM_Exp2_true_R.pdf LAFEM_Exp2_lafem_R.pdf LAFEM_Exp2_Vexpert.pdf LAFEM_Exp2_Vagent.pdf

LAFEM_Exp3:
	make -C InvertedPendulum LAFEM_Exp3_true_R.pdf LAFEM_Exp3_lafem_R.pdf LAFEM_Exp3_Vexpert.pdf LAFEM_Exp3_Vagent.pdf

LAFEM_Exp4:
	make -C InvertedPendulum LAFEM_Exp4_quality.pdf
LAFEM_Exp42:
	make -C InvertedPendulum LAFEM_Exp4_quality_EB.pdf

LAFEM_Exp5:
	make -C InvertedPendulum LAFEM_Exp5_true_R.pdf LAFEM_Exp5_lafem_R.pdf LAFEM_Exp5_Vexpert.pdf LAFEM_Exp5_Vagent.pdf

LAFEM_Exp6:
	make -C Highway FastResults.mat SafeResults.mat

LAFEM_Exp6_EB:
	make -C Highway FastResults_EB.pdf SafeResults_EB.pdf

LAFEM_Exp7:
	make -C InvertedPendulum LAFEM_Exp7_Vphi.pdf LAFEM_Exp7_Vmu.pdf

criteria_mc.tex:
	make -C GridWorld criteria_mc.tex

criteria_lstd.tex:
	make -C GridWorld criteria_lstd.tex

both_error.tex:
	make -C GridWorld both_error.tex

threshold.tex:
	make -C InvertedPendulum threshold.tex

lagoudakis2003least_figure10.pdf:
	make -C ChainWalk lagoudakis2003least_figure10.pdf

test: TT_test0 TT_test1 TT_test2 TT_test3

SUB_DIRS=ChainWalk GridWorld InvertedPendulum  Highway
Makefile:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org &&\
	for dir in $(SUB_DIRS); do $(MAKE) -C $$dir Makefile; done 
