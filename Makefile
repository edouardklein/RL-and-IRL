CFLAGS=-O3 -Wall -pedantic -std=c99 `pkg-config --cflags gsl`
LFLAGS=`pkg-config --libs gsl` -lm -O3

all:

lagoudakis2003least_figure10.pdf: lagoudakis2003least_figure10.ps
	ps2pdf lagoudakis2003least_figure10.ps

lagoudakis2003least_figure10.ps: lagoudakis2003least_figure10.dat lagoudakis2003least_figure10.gp
	gnuplot lagoudakis2003least_figure10.gp

lagoudakis2003least_figure10.dat: lagoudakis2003least_figure10.samples lagoudakis2003least_figure10.exe
	./lagoudakis2003least_figure10.exe > lagoudakis2003least_figure10.dat

lagoudakis2003least_figure10.samples: ChainWalk_generator.exe 
	for i in `seq -w 1000`; do ./ChainWalk_generator.exe 50 > Samples$$i; done && touch lagoudakis2003least_figure10.samples

lagoudakis2003least_figure10.exe: lagoudakis2003least_figure10.o LSPI.o LSTDQ.o utils.o
	gcc -o lagoudakis2003least_figure10.exe lagoudakis2003least_figure10.o LSPI.o LSTDQ.o utils.o $(LFLAGS)

LSPI.o: LSPI.h LSPI.c utils.h LSTDQ.h
	gcc -c $(CFLAGS) LSPI.c

LSTDQ.o: LSTDQ.h LSTDQ.c
	gcc -c $(CFLAGS) LSTDQ.c

utils.o: utils.h utils.c
	gcc -c $(CFLAGS) utils.c

lagoudakis2003least_figure10.o: lagoudakis2003least_figure10.c LSPI.h utils.h
	gcc -c $(CFLAGS) lagoudakis2003least_figure10.c

ChainWalk_generator.exe: ChainWalk_generator.c
	gcc -o ChainWalk_generator.exe $(CFLAGS) $(LFLAGS) ChainWalk_generator.c

clean:
	rm *.o *.exe Samples* *.ps *.pdf *.samples *.dat