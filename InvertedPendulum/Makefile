CFLAGS=-O3 -Wall -pedantic -std=c99 `pkg-config --cflags gsl` -I..
LFLAGS=`pkg-config --libs gsl` -lm -O3

InvertedPendulum.o: InvertedPendulum.c InvertedPendulum.h ../utils.h
	gcc -c $(CFLAGS) InvertedPendulum.c

simulator.o: simulator.c InvertedPendulum.h ../utils.h simulator.h
	gcc -c $(CFLAGS) simulator.c

generator.exe: generator.o ../utils.o InvertedPendulum.o
	gcc -o generator.exe $(LFLAGS) generator.o ../utils.o InvertedPendulum.o

generator.o: generator.c InvertedPendulum.h ../utils.h
	gcc -c $(CFLAGS) generator.c


plot.dat: plot.samples plot.exe
	./plot.exe > plot.dat

clean_plot.dat:
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

plot.samples: generator.exe 
	./generator.exe > Samples.dat && touch plot.samples

plot.exe: plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o InvertedPendulum.o
	gcc -o plot.exe $(LFLAGS) plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o InvertedPendulum.o

plot.o: plot.c InvertedPendulum.h ../utils.h ../LSPI.h ../greedy.h simulator.h ../abbeel2004apprenticeship.h ../LSTDmu.h
	gcc -c $(CFLAGS) plot.c

#Threshold at which LSTDmu starts working for 1 and 100 episodes from the expert
threshold_EB.tex: threshold_EB.pdf
	cat threshold_EB.tex | sed "s|threshold_EB|../Code/InvertedPendulum/threshold_EB|g" > tmp.tex && mv tmp.tex threshold_EB.tex

threshold_EB.pdf: threshold_EB.eps
	epstopdf threshold_EB.eps

threshold_EB.eps: threshold_1.dat-0 threshold_100.dat-0 threshold_EB.gp ErrorBars
	gnuplot threshold_EB.gp

.PHONY: ErrorBars

ErrorBars:
	../../Utils/ErrorBars.py

threshold.tex: threshold.pdf
	cat threshold.tex | sed "s|threshold|../Code/InvertedPendulum/threshold|g" > tmp.tex && mv tmp.tex threshold.tex

threshold.pdf: threshold.eps
	epstopdf threshold.eps

threshold.eps: threshold_1.dat threshold_100.dat threshold.gp
	gnuplot threshold.gp

threshold_1.dat: plot.dat
	cat plot.dat | grep "^1-LSTD" | sed 's/1-LSTD //' | cut -d' ' -f'2 3' > threshold_1.dat

clean_threshold_1.dat: clean_plot.dat

threshold_100.dat: plot.dat
	cat plot.dat | grep "^100-LSTD" | sed 's/100-LSTD //' | cut -d' ' -f'2 3' > threshold_100.dat

clean_threshold_100.dat: clean_plot.dat



clean:
	find . -maxdepth 1 -iname "*.o"       | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.exe"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"        | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

nuke: #You probably don't want to make this one
	find ErrorBars/ -maxdepth 1 -iname "*.dat*" | xargs -tr rm
