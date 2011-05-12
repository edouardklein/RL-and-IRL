
SUB_DIRS=ChainWalk GridWorld InvertedPendulum

lagoudakis2003least_figure10.pdf: LSPI.o LSTDQ.o utils.o greedy.o
	$(MAKE) -C ChainWalk lagoudakis2003least_figure10.pdf && cp ChainWalk/lagoudakis2003least_figure10.pdf ./

both_error_EB.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	$(MAKE) -C GridWorld both_error_EB.tex && cp GridWorld/both_error_EB.pdf ./both_error_EB.pdf

threshold.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	$(MAKE) -C InvertedPendulum threshold.tex && cp InvertedPendulum/threshold.pdf ./threshold.pdf

threshold_EB.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	$(MAKE) -C InvertedPendulum threshold_EB.tex && cp InvertedPendulum/threshold_EB.pdf ./threshold.pdf

criteria_mc.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	$(MAKE) -C GridWorld criteria_mc.tex && cp GridWorld/criteria_mc.pdf ./criteria_mc.pdf

criteria_lstd_EB.pdf: LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o LSTDmu.o criteria.o
	$(MAKE) -C GridWorld criteria_lstd_EB.tex && cp GridWorld/criteria_lstd_EB.pdf ./criteria_lstd_EB.pdf

export CFLAGS=-g -Wall -pedantic -std=c99 `pkg-config --cflags gsl`
export LFLAGS=`pkg-config --libs gsl` -lm -g

ORG_CODE_FILES=LSPI.org RL_Globals.org LSTDQ.org greedy.org Makefile.org ChainWalk/Makefile.org GridWorld/Makefile.org InvertedPendulum/Makefile.org abbeel2004apprenticeship.org criteria.org IRL_Globals.org LSTDmu.org utils.org

HTML_FILES=$(ORG_CODE_FILES:.org=.html)

doc: $(HTML_FILES)
	mkdir -p doc &&\
	for dir in $(SUB_DIRS); do $(MAKE) -C $$dir doc && mkdir -p doc/$$dir && mv $$dir/*.html doc/$$dir/; done &&\
	mv *.html doc/

%.html:%.org
	emacs -batch --visit $*.org --funcall org-export-as-html-and-open --script ~/.emacs

code:$(ORG_CODE_FILES)
	for file in $(ORG_CODE_FILES); do emacs -batch --visit $$file --funcall org-babel-tangle --script ~/.emacs; done &&\
	for dir in $(SUB_DIRS); do $(MAKE) -C $$dir code ; done &&\
	touch code

OBJECT_FILES=LSPI.o LSTDQ.o utils.o greedy.o abbeel2004apprenticeship.o criteria.o LSTDmu.o
obj:$(OBJECT_FILES)

LSPI.o: code LSPI.h LSPI.c utils.h LSTDQ.h greedy.h
	gcc -c $(CFLAGS) LSPI.c

LSTDQ.o: LSTDQ.h LSTDQ.c code
	gcc -c $(CFLAGS) LSTDQ.c

utils.o: utils.h utils.c
	gcc -c $(CFLAGS) utils.c

greedy.o: greedy.h greedy.c code
	gcc -c $(CFLAGS) greedy.c

abbeel2004apprenticeship.o: abbeel2004apprenticeship.c abbeel2004apprenticeship.h LSPI.h utils.h criteria.h
	gcc -c $(CFLAGS) abbeel2004apprenticeship.c

criteria.o: criteria.h criteria.c RL_Globals.h
	gcc -c $(CFLAGS) criteria.c

LSTDmu.o: LSTDmu.h LSTDmu.c greedy.h utils.h criteria.h LSPI.h
	gcc -c $(CFLAGS) LSTDmu.c

clean:
	find . -maxdepth 1 -iname "*.o"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.pdf" | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*~"    | xargs -tr rm &&\
	find . -maxdepth 1 -iname "*.html"    | xargs -tr rm &&\
	$(MAKE) -C ChainWalk clean         
	$(MAKE) -C GridWorld clean
	$(MAKE) -C InvertedPendulum clean
	find . -maxdepth 1 -iname "RL_Globals.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSTDQ.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSTDQ.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSPI.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSPI.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "greedy.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "greedy.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "abbeel2004apprenticeship.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "abbeel2004apprenticeship.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "criteria.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "criteria.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "IRL_Globals.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSTDmu.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "LSTDmu.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "utils.c"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "utils.h"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "code"   | xargs -tr rm &&\
	find . -maxdepth 1 -iname "doc"   | xargs -tr rm -rf
