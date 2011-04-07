CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl` -I..
LFLAGS=`pkg-config --libs gsl` -lm -g

#Figure 10 of \cite{lagoudakis2003least}
lagoudakis2003least_figure10.pdf: lagoudakis2003least_figure10.ps
	ps2pdf lagoudakis2003least_figure10.ps

lagoudakis2003least_figure10.ps: lagoudakis2003least_figure10.dat lagoudakis2003least_figure10.gp
	gnuplot lagoudakis2003least_figure10.gp

lagoudakis2003least_figure10.dat: lagoudakis2003least_figure10.samples lagoudakis2003least_figure10.exe
	./lagoudakis2003least_figure10.exe > lagoudakis2003least_figure10.dat

lagoudakis2003least_figure10.samples: ChainWalk_generator.exe 
	for i in `seq -w 1000`; do ./ChainWalk_generator.exe 50 > Samples$$i; done && touch lagoudakis2003least_figure10.samples

lagoudakis2003least_figure10.exe: lagoudakis2003least_figure10.o ../LSPI.o ../LSTDQ.o ../utils.o ../greedy.o 
	gcc -o lagoudakis2003least_figure10.exe lagoudakis2003least_figure10.o ../LSPI.o ../LSTDQ.o ../utils.o ../greedy.o  $(LFLAGS)

lagoudakis2003least_figure10.o: lagoudakis2003least_figure10.c ../LSPI.h ../utils.h
	gcc -c $(CFLAGS) lagoudakis2003least_figure10.c

ChainWalk_generator.exe: ChainWalk_generator.o ../utils.o
	gcc -o ChainWalk_generator.exe  $(LFLAGS) ChainWalk_generator.o ../utils.o

ChainWalk_generator.o: ChainWalk_generator.c ../utils.h
	gcc -c $(CFLAGS) ChainWalk_generator.c

clean: 
	find . -maxdepth 1 -iname "*.o"       | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.ps"      | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.exe"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"        | xargs -tr rm &&\
	find . -maxdepth 1 -iname "Samples*"  | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

