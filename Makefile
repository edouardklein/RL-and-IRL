CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl`
LFLAGS=`pkg-config --libs gsl` -lm -g

all: lagoudakis2003least_figure10.pdf both_error_discrete_EB.pdf criteria_discrete_lstd_EB.pdf criteria_discrete_mc.pdf 

#Figure 10 of \cite{lagoudakis2003least}
lagoudakis2003least_figure10.pdf: LSPI.o LSTDQ.o utils.o greedy.o
	make -C ChainWalk lagoudakis2003least_figure10.pdf && cp ChainWalk/lagoudakis2003least_figure10.pdf ./

#Different criteria on the gridworld for Monte_Carlo w.r.t. iterations
criteria_discrete_mc.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	make -C GridWorld criteria_mc.tex && cp GridWorld/criteria_mc.pdf ./criteria_discrete_mc.pdf

#Different criteria for LSTD w.r.t. number of samples from the expert
criteria_discrete_lstd.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	make -C GridWorld criteria_lstd.pdf && cp GridWorld/criteria_lstd.pdf ./criteria_discrete_lstd.pdf

#True error for LSTD and MC on the GridWorld
both_error_discrete.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	make -C GridWorld both_error.pdf && cp GridWorld/both_error.pdf ./both_error_discrete.pdf

#True error for LSTD and MC on the GridWorld, with error bars
both_error_discrete_EB.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	make -C GridWorld both_error_EB.tex && cp GridWorld/both_error_EB.pdf ./both_error_discrete_EB.pdf

#Different criteria for LSTD w.r.t. number of samples from the expert, with error bars
criteria_discrete_lstd_EB.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	make -C GridWorld criteria_lstd_EB.tex && cp GridWorld/criteria_lstd_EB.pdf ./criteria_discrete_lstd_EB.pdf


#Common to all
LSPI.o: LSPI.h LSPI.c utils.h LSTDQ.h greedy.h 
	gcc -c $(CFLAGS) LSPI.c

LSTDQ.o: LSTDQ.h LSTDQ.c
	gcc -c $(CFLAGS) LSTDQ.c

utils.o: utils.h utils.c
	gcc -c $(CFLAGS) utils.c

greedy.o: greedy.h greedy.c
	gcc -c $(CFLAGS) greedy.c

#Common for LSTDMu
InvertedPendulum.o: InvertedPendulum.c InvertedPendulum.h utils.h
	gcc -c $(CFLAGS) InvertedPendulum.c

InvertedPendulum_simulator.o: InvertedPendulum_simulator.c InvertedPendulum.h utils.h
	gcc -c $(CFLAGS) InvertedPendulum_simulator.c 

InvertedPendulum_generator.exe: InvertedPendulum_generator.o utils.o InvertedPendulum.o
	gcc -o InvertedPendulum_generator.exe $(LFLAGS) InvertedPendulum_generator.o utils.o InvertedPendulum.o

InvertedPendulum_generator.o: InvertedPendulum_generator.c InvertedPendulum.h utils.h 
	gcc -c $(CFLAGS) InvertedPendulum_generator.c

abbeel2004apprenticeship.o: abbeel2004apprenticeship.c abbeel2004apprenticeship.h LSPI.h utils.h criteria.h
	gcc -c $(CFLAGS) abbeel2004apprenticeship.c

criteria.o: criteria.h criteria.c RL_Globals.h
	gcc -c $(CFLAGS) criteria.c

courbe_lstdmu_continuous.samples: InvertedPendulum_generator.exe 
	./InvertedPendulum_generator.exe > SamplesC.dat && touch courbe_lstdmu_countinuous.samples

courbe_lstdmu_continuous.exe: courbe_lstdmu_continuous.o utils.o LSPI.o InvertedPendulum_simulator.o InvertedPendulum.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	gcc -o courbe_lstdmu_continuous.exe $(LFLAGS) courbe_lstdmu_continuous.o utils.o LSPI.o InvertedPendulum_simulator.o InvertedPendulum.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o criteria.o 

courbe_lstdmu_continuous.o: courbe_lstdmu_continuous.c InvertedPendulum.h utils.h LSPI.h greedy.h GridWorld_simulator.h abbeel2004apprenticeship.h LSTDmu.h
	gcc -c $(CFLAGS) courbe_lstdmu_continuous.c

LSTDmu.o: LSTDmu.h LSTDmu.c greedy.h utils.h criteria.h LSPI.h
	gcc -c $(CFLAGS) LSTDmu.c

clean:
	find . -maxdepth 1 -iname "*.o"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf" | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"    | xargs -tr rm &&\
	make -C ChainWalk clean		
	make -C GridWorld clean
