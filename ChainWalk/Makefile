
lagoudakis2003least_figure10.pdf: lagoudakis2003least_figure10.ps
	ps2pdf lagoudakis2003least_figure10.ps

lagoudakis2003least_figure10.ps: lagoudakis2003least_figure10.dat lagoudakis2003least_figure10.gp
	gnuplot lagoudakis2003least_figure10.gp

lagoudakis2003least_figure10.dat: lagoudakis2003least_figure10.samples lagoudakis2003least_figure10.exe
	./lagoudakis2003least_figure10.exe > lagoudakis2003least_figure10.dat

lagoudakis2003least_figure10.samples: ChainWalk_generator.exe 
	for i in `seq -w 1000`; do ./ChainWalk_generator.exe 50 > Samples$$i; done && touch lagoudakis2003least_figure10.samples

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs

ChainWalk_generator.c: ChainWalk_generator.org 
	$(call tangle,"ChainWalk_generator.org")

lagoudakis2003least_figure10.c: lagoudakis2003least_figure10.org 
	$(call tangle,"lagoudakis2003least_figure10.org")
lagoudakis2003least_figure10.gp: lagoudakis2003least_figure10.org 
	$(call tangle,"lagoudakis2003least_figure10.org")

../utils.h:
	make -C .. utils.h

ChainWalk_generator.o: ChainWalk_generator.c ../utils.h 
	$(call c2obj,"ChainWalk_generator.c")

../LSPI.h:
	make -C .. LSPI.h

lagoudakis2003least_figure10.o: lagoudakis2003least_figure10.c ../LSPI.h ../utils.h
	$(call c2obj,"lagoudakis2003least_figure10.c")

O2EXE=gcc $(LFLAGS)
ChainWalk_generator.exe: ChainWalk_generator.o ../utils.o
	$(O2EXE) -o ChainWalk_generator.exe ChainWalk_generator.o ../utils.o

../LSPI.o:
	make -C .. LSPI.o

../LSTDQ.o:
	make -C .. LSTDQ.o

../utils.o:
	make -C .. utils.o

../greedy.o:
	make -C .. greedy.o

lagoudakis2003least_figure10.exe: lagoudakis2003least_figure10.o ../LSPI.o ../LSTDQ.o ../utils.o ../greedy.o 
	$(O2EXE) -o lagoudakis2003least_figure10.exe lagoudakis2003least_figure10.o ../LSPI.o ../LSTDQ.o ../utils.o ../greedy.o  $(LFLAGS)
generator.exe: generator.o ../utils.o

Generator_clean:
	find . -maxdepth 1 -iname "Chainwalk_generator.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Chainwalk_generator.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "Chainwalk_generator.o"   | xargs $(XARGS_OPT) rm

lagoudakis2003least_figure10_clean:
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.c"   | xargs $(XARGS_OPT) rm 
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.gp"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.ps"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "lagoudakis2003least_figure10.samples"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Samples*"   | xargs $(XARGS_OPT) rm

clean: Generator_clean lagoudakis2003least_figure10_clean

Makefile:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org

  CFLAGS+=-I..
c2obj=gcc $(CFLAGS) -c $1
