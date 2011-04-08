CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl` -I..
LFLAGS=`pkg-config --libs gsl` -lm -g

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

#Criteria for Monte-Carlo w.r.t. Iterations
criteria_mc.pdf: criteria_mc.eps
	epstopdf criteria_mc.eps

criteria_mc.eps: criteria_mc.dat criteria_mc.gp
	gnuplot criteria_mc.gp

criteria_mc.dat: plot.dat
	cat plot.dat | grep "^1000" | sed 's/1000 //'> criteria_mc.dat

clean_criteria_mc.dat: clean_plot.dat
	rm criteria_mc.dat

# #Criteria for LSTD w.r.t. number of samples from the expert
# criteria_lstd_EB.pdf: criteria_lstd_EB.eps
# 	epstopdf criteria_lstd_EB.eps

# criteria_lstd_EB.eps: criteria_lstd.dat-0 criteria_lstd.dat-1 criteria_lstd.dat-2 criteria_lstd.dat-3 criteria_lstd_EB.gp
# 	gnuplot criteria_lstd_EB.gp

criteria_lstd.pdf: criteria_lstd.eps
	epstopdf criteria_lstd.eps

criteria_lstd.eps: criteria_lstd.dat criteria_lstd.gp
	gnuplot criteria_lstd.gp

criteria_lstd.dat: plot.dat
	cat plot.dat | grep -E "^B " | sed "s/B //"> criteria_lstd.dat

clean_criteria_lstd.dat: clean_plot.dat
	rm criteria_lstd.dat

#Criteria for MC w.r.t. number of samples from the expert
criteria_mc_samples.dat: plot.dat
	cat plot.dat | grep -E "^C " | sed "s/C //"> criteria_mc_samples.dat

clean_mc_samples.dat: clean_plot.dat
	rm criteria_mc_samples.dat

# #True error for both MC and LSTD w.r.t. number of samples from the expert
# both_error_EB.pdf: both_error_EB.eps
# 	epstopdf both_error_EB.eps

# both_error_EB.eps: both_error.dat-0 both_error.dat-1 both_error_EB.gp
# 	gnuplot both_error_EB.gp

# both_error.dat-0: ErrorBars

# both_errot.dat-1: ErrorBars

# ErrorBars: conf.py
# 	../../Utils/ErrorBars.py

both_error.pdf: both_error.eps
	epstopdf both_error.eps

both_error.eps: both_error.dat both_error.gp
	gnuplot both_error.gp

both_error.dat: criteria_mc_samples.dat criteria_lstd.dat
	cat criteria_lstd.dat | cut -d" " -f'1 4' > tmp &&\
	cat criteria_mc_samples.dat | cut -d" " -f'1 4' > tmp2 &&\
	paste -d' ' tmp tmp2 > both_error.dat

clean_both_error.dat: clean_criteria_mc_samples.dat clean_criteria_lstd.dat
	rm both_error.dat

clean:
	find . -maxdepth 1 -iname "*.o"       | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.exe"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.dat"     | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"        | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.samples" | xargs -tr rm

# nuke: #You probably don't want to make this one
# 	find ErrorBars/ -maxdepth 1 -iname "plot.dat*" | xargs -tr rm
