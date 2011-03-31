CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl`
LFLAGS=`pkg-config --libs gsl` -lm -g

all:

#Figure 10 of \cite{lagoudakis2003least}
lagoudakis2003least_figure10.pdf: lagoudakis2003least_figure10.ps
	ps2pdf lagoudakis2003least_figure10.ps

lagoudakis2003least_figure10.ps: lagoudakis2003least_figure10.dat lagoudakis2003least_figure10.gp
	gnuplot lagoudakis2003least_figure10.gp

lagoudakis2003least_figure10.dat: lagoudakis2003least_figure10.samples lagoudakis2003least_figure10.exe
	./lagoudakis2003least_figure10.exe > lagoudakis2003least_figure10.dat

lagoudakis2003least_figure10.samples: ChainWalk_generator.exe 
	for i in `seq -w 1000`; do ./ChainWalk_generator.exe 50 > Samples$$i; done && touch lagoudakis2003least_figure10.samples

lagoudakis2003least_figure10.exe: lagoudakis2003least_figure10.o LSPI.o LSTDQ.o utils.o greedy.o 
	gcc -o lagoudakis2003least_figure10.exe lagoudakis2003least_figure10.o LSPI.o LSTDQ.o utils.o greedy.o  $(LFLAGS)

lagoudakis2003least_figure10.o: lagoudakis2003least_figure10.c LSPI.h utils.h
	gcc -c $(CFLAGS) lagoudakis2003least_figure10.c

ChainWalk_generator.exe: ChainWalk_generator.o utils.o
	gcc -o ChainWalk_generator.exe  $(LFLAGS) ChainWalk_generator.o utils.o

ChainWalk_generator.o: ChainWalk_generator.c utils.h
	gcc -c $(CFLAGS) ChainWalk_generator.c

#Common
LSPI.o: LSPI.h LSPI.c utils.h LSTDQ.h greedy.h 
	gcc -c $(CFLAGS) LSPI.c

LSTDQ.o: LSTDQ.h LSTDQ.c
	gcc -c $(CFLAGS) LSTDQ.c

utils.o: utils.h utils.c
	gcc -c $(CFLAGS) utils.c

greedy.o: greedy.h greedy.c
	gcc -c $(CFLAGS) greedy.c

#Common for LSTDMu
GridWorld_simulator.o: GridWorld_simulator.c GridWorld.h utils.h
	gcc -c $(CFLAGS) GridWorld_simulator.c

GridWorld_generator.exe: GridWorld_generator.o utils.o
	gcc -o GridWorld_generator.exe $(LFLAGS) GridWorld_generator.o utils.o

GridWorld_generator.o: GridWorld_generator.c GridWorld.h utils.h
	gcc -c $(CFLAGS) GridWorld_generator.c

abbeel2004apprenticeship.o: abbeel2004apprenticeship.c abbeel2004apprenticeship.h LSPI.h utils.h criteria.h
	gcc -c $(CFLAGS) abbeel2004apprenticeship.c

criteria.o: criteria.h criteria.c RL_Globals.h
	gcc -c $(CFLAGS) criteria.c

courbe_lstdmu.dat: courbe_lstdmu.samples courbe_lstdmu.exe
	./courbe_lstdmu.exe | tee courbe_lstdmu.dat

courbe_lstdmu.samples: GridWorld_generator.exe 
	./GridWorld_generator.exe > Samples.dat && touch courbe_lstdmu.samples

courbe_lstdmu.exe: courbe_lstdmu.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	gcc -o courbe_lstdmu.exe $(LFLAGS) courbe_lstdmu.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o criteria.o

courbe_lstdmu.o: courbe_lstdmu.c GridWorld.h utils.h LSPI.h greedy.h GridWorld_simulator.h abbeel2004apprenticeship.h LSTDmu.h
	gcc -c $(CFLAGS) courbe_lstdmu.c

LSTDmu.o: LSTDmu.h LSTDmu.c greedy.h utils.h criteria.h LSPI.h
	gcc -c $(CFLAGS) LSTDmu.c

#Courbe A :
courbe_a.pdf: courbe_a.ps
	ps2pdf courbe_a.ps

courbe_a.ps: courbe_a.dat courbe_a.gp
	gnuplot courbe_a.gp

courbe_a.dat: courbe_lstdmu.dat
	cat courbe_lstdmu.dat | grep "^500" > courbe_a.dat

#Courbe B : 
courbe_b.pdf: courbe_b.ps
	ps2pdf courbe_b.ps

courbe_b.ps: courbe_b.dat courbe_b.gp
	gnuplot courbe_b.gp

courbe_b.dat: courbe_lstdmu.dat
	cat courbe_lstdmu.dat | grep -E "^0 1" > courbe_b.dat

#Courbe B : 
courbe_c.pdf: courbe_c.ps
	ps2pdf courbe_c.ps

courbe_c.ps: courbe_c.dat courbe_c.gp
	gnuplot courbe_c.gp

courbe_c.dat: courbe_lstdmu.dat
	cat courbe_lstdmu.dat | grep -E "^C " | sed "s/C//"> courbe_c.dat
#Courbe D :
courbe_d.pdf: courbe_d.ps
	ps2pdf courbe_d.ps

courbe_d.ps: courbe_b.dat courbe_c.dat courbe_d.gp
	gnuplot courbe_d.gp
#Courbe E :
courbe_e.pdf: courbe_e.ps
	ps2pdf courbe_e.ps

courbe_e.ps: courbe_b.dat courbe_c.dat courbe_e.gp
	gnuplot courbe_e.gp

clean:
	rm *.o *.exe Samples* *.ps *.pdf *.samples *.dat
