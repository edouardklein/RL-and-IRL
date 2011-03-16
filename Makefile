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
	gcc -o lagoudakis2003least_figure10.exe lagoudakis2003least_figure10.o LSPI.o LSTDQ.o utils.o greedy.o $(LFLAGS)

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

#Common to curves A and B
GridWorld_simulator.o: GridWorld_simulator.c GridWorld.h utils.h
	gcc -c $(CFLAGS) GridWorld_simulator.c

GridWorld_generator.exe: GridWorld_generator.o utils.o
	gcc -o GridWorld_generator.exe $(LFLAGS) GridWorld_generator.o utils.o

GridWorld_generator.o: GridWorld_generator.c GridWorld.h utils.h
	gcc -c $(CFLAGS) GridWorld_generator.c

abbeel2004apprenticeship.o: abbeel2004apprenticeship.c abbeel2004apprenticeship.h LSPI.h utils.h
	gcc -c $(CFLAGS) abbeel2004apprenticeship.c

#Courbe A :
courbe_a.pdf: courbe_a.ps
	ps2pdf courbe_a.ps

courbe_a.ps: courbe_a.dat courbe_a.gp
	gnuplot courbe_a.gp

courbe_a.dat: courbe_a.samples courbe_a.exe
	./courbe_a.exe > courbe_a.dat

courbe_a.samples: GridWorld_generator.exe 
	./GridWorld_generator.exe > Samples.dat && touch courbe_a.samples

courbe_a.exe: courbe_a.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o
	gcc -o courbe_a.exe $(LFLAGS) courbe_a.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o

courbe_a.o: courbe_a.c GridWorld.h utils.h LSPI.h greedy.h GridWorld_simulator.h abbeel2004apprenticeship.h
	gcc -c $(CFLAGS) courbe_a.c

#Courbe B : 
courbe_b.pdf: courbe_b.ps
	ps2pdf courbe_b.ps

courbe_b.ps: courbe_b_50.dat courbe_b_100.dat courbe_b_150.dat courbe_b_Nous.dat courbe_b.gp
	gnuplot courbe_b.gp

courbe_b_50.dat: courbe_b.dat
	cat courbe_b.dat | grep -E "^5" > courbe_b_50.dat

courbe_b_100.dat: courbe_b.dat
	cat courbe_b.dat | grep -E "^20" > courbe_b_100.dat

courbe_b_150.dat: courbe_b.dat
	cat courbe_b.dat | grep -E "^35" > courbe_b_150.dat

courbe_b_Nous.dat: courbe_b.dat
	cat courbe_b.dat | grep -E "^0" | grep -E "000$$" > courbe_b_Nous.dat

courbe_b.dat: courbe_b.samples courbe_b.exe
	./courbe_b.exe > courbe_b.dat

courbe_b.samples: GridWorld_generator.exe 
	./GridWorld_generator.exe > Samples.dat && touch courbe_b.samples

courbe_b.exe: courbe_b.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o
	gcc -o courbe_b.exe $(LFLAGS) courbe_b.o utils.o LSPI.o GridWorld_simulator.o greedy.o LSTDQ.o abbeel2004apprenticeship.o LSTDmu.o

courbe_b.o: courbe_b.c GridWorld.h utils.h LSPI.h greedy.h GridWorld_simulator.h abbeel2004apprenticeship.h LSTDmu.h
	gcc -c $(CFLAGS) courbe_b.c

LSTDmu.o: LSTDmu.h LSTDmu.c
	gcc -c $(CFLAGS) LSTDmu.c



clean:
	rm *.o *.exe Samples* *.ps *.pdf *.samples *.dat
