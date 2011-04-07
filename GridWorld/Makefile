CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl` -I..
LFLAGS=`pkg-config --libs gsl` -lm -g

simulator.o: simulator.c GridWorld.h ../utils.h
	gcc -c $(CFLAGS) simulator.c

generator.exe: generator.o ../utils.o
	gcc -o generator.exe $(LFLAGS) generator.o ../utils.o

generator.o: generator.c GridWorld.h ../utils.h
	gcc -c $(CFLAGS) generator.c

plot.dat: plot.samples plot.exe
	./plot.exe > plot.dat

clean_plot.dat:
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

plot.samples: generator.exe 
	./generator.exe > Samples.dat && touch plot.samples

plot.exe: plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o
	gcc -o plot.exe $(LFLAGS) plot.o ../utils.o ../LSPI.o simulator.o ../greedy.o ../LSTDQ.o ../abbeel2004apprenticeship.o ../LSTDmu.o ../criteria.o

plot.o: plot.c GridWorld.h ../utils.h ../LSPI.h ../greedy.h simulator.h ../abbeel2004apprenticeship.h ../LSTDmu.h
	gcc -c $(CFLAGS) plot.c

#Criteria for Monte-Carlo w.r.t. Iterations
criteria_mc.tex: criteria_mc.pdf
	cat criteria_mc.tex | sed "s|criteria_mc|../Code/GridWorld/criteria_mc|g" > tmp.tex && mv tmp.tex criteria_mc.tex

criteria_mc.pdf: criteria_mc.eps
	epstopdf criteria_mc.eps

criteria_mc.eps: criteria_mc.dat criteria_mc.gp
	gnuplot criteria_mc.gp

criteria_mc.dat: plot.dat
	cat plot.dat | grep "^500" | sed 's/500 //'> criteria_mc.dat

clean_criteria_mc.dat: clean_plot.dat
	rm criteria_mc.dat

#Criteria for LSTD w.r.t. number of samples from the expert
criteria_lstd_EB.tex: criteria_lstd_EB.pdf
	cat criteria_lstd_EB.tex | sed "s|criteria_lstd_EB|../Code/GridWorld/criteria_lstd_EB|g" > tmp.tex && mv tmp.tex criteria_lstd_EB.tex

criteria_lstd_EB.pdf: criteria_lstd_EB.eps
	epstopdf criteria_lstd_EB.eps

criteria_lstd_EB.eps: criteria_lstd.dat-0 criteria_lstd.dat-1 criteria_lstd.dat-2 criteria_lstd.dat-3 criteria_lstd_EB.gp
	gnuplot criteria_lstd_EB.gp

criteria_lstd.pdf: criteria_lstd.eps
	epstopdf criteria_lstd.eps

criteria_lstd.eps: criteria_lstd.dat criteria_lstd.gp
	gnuplot criteria_lstd.gp

criteria_lstd.dat: plot.dat
	cat plot.dat | grep -E "^B " | sed "s/B //"> criteria_lstd.dat

clean_criteria_lstd.dat: clean_plot.dat

#Criteria for MC w.r.t. number of samples from the expert
criteria_mc_samples.dat: plot.dat
	cat plot.dat | grep -E "^C " | sed "s/C //"> criteria_mc_samples.dat

clean_criteria_mc_samples.dat: clean_plot.dat

#True error for both MC and LSTD w.r.t. number of samples from the expert
both_error_EB.tex: both_error_EB.pdf
	cat both_error_EB.tex | sed "s|both_error_EB|../Code/GridWorld/both_error_EB|g" > tmp.tex && mv tmp.tex both_error_EB.tex

both_error_EB.pdf: both_error_EB.eps
	epstopdf both_error_EB.eps

both_error_EB.eps: both_error.dat-0 both_error.dat-1 both_error_EB.gp
	gnuplot both_error_EB.gp

both_error.dat-0: ErrorBars

both_errot.dat-1: ErrorBars

ErrorBars: conf.py
	../../Utils/ErrorBars.py

both_error.pdf: both_error.eps
	epstopdf both_error.eps

both_error.eps: both_error.dat both_error.gp
	gnuplot both_error.gp

both_error.dat: criteria_mc_samples.dat criteria_lstd.dat
	cat criteria_lstd.dat | cut -d" " -f'1 4' > tmp &&\
	cat criteria_mc_samples.dat | cut -d" " -f'1 4' > tmp2 &&\
	paste -d' ' tmp tmp2 > both_error.dat

clean_both_error.dat: clean_criteria_mc_samples.dat clean_criteria_lstd.dat

clean:
	find . -maxdepth 1 -iname "*.o"       | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.eps"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.ps"      | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.tex"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.exe"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.dat-*"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pyc"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"        | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

nuke: #You probably don't want to make this one
	find ErrorBars/ -maxdepth 1 -iname "plot.dat*" | xargs -tr rm
