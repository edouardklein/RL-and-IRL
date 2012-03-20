
phipsi.py: Main.org
	$(call tangle,"Main.org")

phipsi.h: Main.org
	$(call tangle,"Main.org")
phipsi.c: Main.org
	$(call tangle,"Main.org")

phipsi.o: phipsi.c phipsi.h
	$(call c2obj,"phipsi.c")

al_code/fast_reward.mat: MWAL.org
	$(call tangle,"MWAL.org")
al_code/safe_reward.mat: MWAL.org
	$(call tangle,"MWAL.org")

al_code/fast_policy.dat: al_code/make_THETA.m al_code/make_F.m al_code/optimal_policy.m al_code/fast_reward.mat
	pushd al_code; octave optimal_policy.m fast_reward.mat ; mv policy.dat fast_policy.dat ; popd
al_code/safe_policy.dat: al_code/make_THETA.m al_code/make_F.m al_code/optimal_policy.m al_code/safe_reward.mat
	pushd al_code; octave optimal_policy.m safe_reward.mat ; mv policy.dat safe_policy.dat ; popd

al_code/fast_policy.txt: al_code/fast_policy.dat
	pushd al_code; mv fast_policy.dat policy.dat; perl post.pl ;mv policy.txt fast_policy.txt; mv policy.dat fast_policy.dat ; popd
al_code/safe_policy.txt: al_code/safe_policy.dat
	pushd al_code; mv safe_policy.dat policy.dat; perl post.pl ;mv policy.txt safe_policy.txt; mv policy.dat safe_policy.dat ; popd

PlayFast.dat : al_code/fast_policy.txt PolicyPlay.py App.py
	python PolicyPlay.py al_code/fast_policy.txt > PlayFast.dat

PlaySafe.dat: al_code/safe_policy.txt PolicyPlay.py App.py
	python PolicyPlay.py al_code/safe_policy.txt > PlaySafe.dat

PlayFast_stripped.dat : PlayFast.dat
	cat PlayFast.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlayFast_stripped.dat

PlaySafe_stripped.dat : PlaySafe.dat
	cat PlaySafe.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlaySafe_stripped.dat

N_VALUES=30 100 150 300

PlayFast_X.dat: PlayFast_stripped.dat #Dummy target
	for i in $(N_VALUES); do head -n $$i PlayFast_stripped.dat > PlayFast_$$i.dat; done
	touch PlayFast_X.dat

PlaySafe_X.dat: PlaySafe_stripped.dat #Dummy target
	for i in $(N_VALUES); do head -n $$i PlaySafe_stripped.dat > PlaySafe_$$i.dat; done
	touch PlaySafe_X.dat

tr2mu.py: MWAL.org
	$(call tangle,"MWAL.org")

PlayFast_mu_X.mat : PlayFast_X.dat tr2mu.py phipsi.py
	for i in $(N_VALUES); do python tr2mu.py PlayFast_$$i.dat | tr -d '[' | tr -d ']' > PlayFast_mu_$$i.mat; done
	touch PlayFast_mu_X.mat

PlaySafe_mu_X.mat : PlaySafe_X.dat tr2mu.py phipsi.py
	for i in $(N_VALUES); do python tr2mu.py PlaySafe_$$i.dat | tr -d '[' | tr -d ']' > PlaySafe_mu_$$i.mat; done
	touch PlaySafe_mu_X.mat

PlayFast_mwal_mu_X.mat : PlayFast_mu_X.mat al_code/my_MWAL.m
	for i in $(N_VALUES); do \
	pushd al_code; octave my_MWAL.m ../PlayFast_mu_$$i.mat; perl post.pl ; mv policy.txt policy_mwal_fast_$$i.txt ; popd;\
	python PolicyPlay.py al_code/policy_mwal_fast_$$i.txt > PlayFast_mwal_$$i.dat;\
	cat PlayFast_mwal_$$i.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlayFast_mwal_stripped_$$i.dat;\
	python tr2mu.py PlayFast_mwal_stripped_$$i.dat | tr -d '[' | tr -d ']' > PlayFast_mwal_mu_$$i.mat; done
	touch PlayFast_mwal_mu_X.mat

PlaySafe_mwal_mu_X.mat : PlaySafe_mu_X.mat al_code/my_MWAL.m
	for i in $(N_VALUES); do \
	pushd al_code; octave my_MWAL.m ../PlaySafe_mu_$$i.mat; perl post.pl ; mv policy.txt policy_mwal_safe_$$i.txt ; popd;\
	python PolicyPlay.py al_code/policy_mwal_safe_$$i.txt > PlaySafe_mwal_$$i.dat;\
	cat PlaySafe_mwal_$$i.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlaySafe_mwal_stripped_$$i.dat;\
	python tr2mu.py PlaySafe_mwal_stripped_$$i.dat | tr -d '[' | tr -d ']' > PlaySafe_mwal_mu_$$i.mat; done
	touch PlaySafe_mwal_mu_X.mat


sa2saspsie.py: SCIRL.org
	$(call tangle, "SCIRL.org")

PlayFast_saspsie_X.mat : PlayFast_X.dat sa2saspsie.py phipsi.py
	for i in $(N_VALUES); do python sa2saspsie.py PlayFast_$$i.dat PlayFast_saspsie_$$i.mat ; done
	touch PlayFast_saspsie_X.mat

PlaySafe_saspsie_X.mat : PlaySafe_X.dat sa2saspsie.py phipsi.py
	for i in $(N_VALUES); do python sa2saspsie.py PlaySafe_$$i.dat PlaySafe_saspsie_$$i.mat ; done
	touch PlaySafe_saspsie_X.mat

SCIRL_LSTDmu.exe: SCIRL_LSTDmu.o phipsi.o ../utils.o ../LSTDmu.o
	$(O2EXE) -o SCIRL_LSTDmu.exe SCIRL_LSTDmu.o phipsi.o ../utils.o ../LSTDmu.o

SCIRL_LSTDmu.o: SCIRL_LSTDmu.c ../LSTDmu.h ../utils.h ../RL_Globals.h ../IRL_Globals.h phipsi.h
	$(call c2obj,"SCIRL_LSTDmu.c")

SCIRL_LSTDmu.c: SCIRL.org
	$(call tangle,"SCIRL.org")

omega_mu_E_Fast_X.mat: SCIRL_LSTDmu.exe PlayFast_saspsie_X.mat
	for i in $(N_VALUES); do ./SCIRL_LSTDmu.exe PlayFast_saspsie_$$i.mat > omega_mu_E_Fast_$$i.mat; done
	touch omega_mu_E_Fast_X.mat

omega_mu_E_Safe_X.mat: SCIRL_LSTDmu.exe PlaySafe_saspsie_X.mat
	for i in $(N_VALUES); do ./SCIRL_LSTDmu.exe PlaySafe_saspsie_$$i.mat > omega_mu_E_Safe_$$i.mat; done
	touch omega_mu_E_Safe_X.mat

../LAFEM.py:
	make -C .. LAFEM.py

../a2str.py:
	make -C .. a2str.py

SCIRL_Exp6.py: SCIRL.org
	$(call tangle,"SCIRL.org")

theta_lafem_Fast_X.mat: SCIRL_Exp6.py ../a2str.py ../LAFEM.py omega_mu_E_Fast_X.mat phipsi.py PlayFast_X.dat
	for i in $(N_VALUES); do python SCIRL_Exp6.py PlayFast_$$i.dat omega_mu_E_Fast_$$i.mat > theta_lafem_Fast_$$i.mat; done
	touch theta_lafem_Fast_X.mat

theta_lafem_Safe_X.mat: SCIRL_Exp6.py ../a2str.py ../LAFEM.py omega_mu_E_Safe_X.mat phipsi.py PlaySafe_X.dat
	for i in $(N_VALUES); do python SCIRL_Exp6.py PlaySafe_$$i.dat omega_mu_E_Safe_$$i.mat > theta_lafem_Safe_$$i.mat; done
	touch theta_lafem_Safe_X.mat

al_code/optimal_policy.m: SCIRL.org
	$(call tangle,"SCIRL.org")

al_code/make_F.m: al_code/pre.pl
	pushd al_code; perl pre.pl ; popd

al_code/make_THETA.m: al_code/pre.pl
	pushd al_code; perl pre.pl ; popd

al_code/scirl_fast_policy_X.txt: al_code/make_THETA.m al_code/make_F.m al_code/optimal_policy.m theta_lafem_Fast_X.mat
	for i in $(N_VALUES); do pushd al_code; octave optimal_policy.m ../theta_lafem_Fast_$$i.mat ; perl post.pl ; mv policy.txt scirl_fast_policy_$$i.txt ; popd; done
	touch al_code/scirl_fast_policy_X.txt

al_code/scirl_safe_policy_X.txt: al_code/make_THETA.m al_code/make_F.m al_code/optimal_policy.m theta_lafem_Safe_X.mat
	for i in $(N_VALUES); do pushd al_code; octave optimal_policy.m ../theta_lafem_Safe_$$i.mat ; perl post.pl ; mv policy.txt scirl_safe_policy_$$i.txt ; popd; done
	touch al_code/scirl_safe_policy_X.txt

PlayFast_scirl_mu_X.mat :al_code/scirl_fast_policy_X.txt tr2mu.py
	for i in $(N_VALUES); do \
	python PolicyPlay.py al_code/scirl_fast_policy_$$i.txt > PlayFast_scirl_$$i.dat;\
	cat PlayFast_scirl_$$i.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlayFast_scirl_stripped_$$i.dat;\
	python tr2mu.py PlayFast_scirl_stripped_$$i.dat | tr -d '[' | tr -d ']' > PlayFast_scirl_mu_$$i.mat; done
	touch PlayFast_scirl_mu_X.mat

PlaySafe_scirl_mu_X.mat :al_code/scirl_safe_policy_X.txt tr2mu.py
	for i in $(N_VALUES); do \
	python PolicyPlay.py al_code/scirl_safe_policy_$$i.txt > PlaySafe_scirl_$$i.dat;\
	cat PlaySafe_scirl_$$i.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlaySafe_scirl_stripped_$$i.dat;\
	python tr2mu.py PlaySafe_scirl_stripped_$$i.dat | tr -d '[' | tr -d ']' > PlaySafe_scirl_mu_$$i.mat; done
	touch PlaySafe_scirl_mu_X.mat


FastResults_scirl.py: Comparison.org
	$(call tangle,"Comparison.org")
FastResults_mwal.py: Comparison.org
	$(call tangle,"Comparison.org")
SafeResults_scirl.py: Comparison.org
	$(call tangle,"Comparison.org")
SafeResults_mwal.py: Comparison.org
	$(call tangle,"Comparison.org")
ExpertsResults.py: Comparison.org
	$(call tangle,"Comparison.org")

FastResults_scirl.mat: FastResults_scirl.py PlayFast_scirl_mu_X.mat al_code/fast_reward.mat
	touch FastResults_scirl.mat
	rm FastResults_scirl.mat
	for i in $(N_VALUES); do (/bin/echo -n "$$i " && python FastResults_scirl.py $$i) >> FastResults_scirl.mat; done

FastResults_mwal.mat: FastResults_mwal.py PlayFast_mwal_mu_X.mat al_code/fast_reward.mat
	touch FastResults_mwal.mat
	rm FastResults_mwal.mat
	for i in $(N_VALUES); do (/bin/echo -n "$$i " && python FastResults_mwal.py $$i) >> FastResults_mwal.mat; done

SafeResults_scirl.mat: SafeResults_scirl.py PlaySafe_scirl_mu_X.mat al_code/safe_reward.mat
	touch SafeResults_scirl.mat
	rm SafeResults_scirl.mat
	for i in $(N_VALUES); do (/bin/echo -n "$$i " && python SafeResults_scirl.py $$i) >> SafeResults_scirl.mat; done

SafeResults_mwal.mat: SafeResults_mwal.py PlaySafe_mwal_mu_X.mat al_code/safe_reward.mat
	touch SafeResults_mwal.mat
	rm SafeResults_mwal.mat
	for i in $(N_VALUES); do (/bin/echo -n "$$i " && python SafeResults_mwal.py $$i) >> SafeResults_mwal.mat; done

#FIXME:On devrauit pas avoir à spécifier 300 ici, si on change N_VALUES, on est potentiellement marrons
SafeResults_true.mat: PlaySafe_mu_X.mat al_code/safe_reward.mat PlayFast_mu_X.mat
	$(call tangle,"Comparison.org")
	python ExpertsResults.py 

FastResults_true.mat: PlayFast_mu_X.mat al_code/fast_reward.mat PlaySafe_mu_X.mat
	$(call tangle,"Comparison.org")
	python ExpertsResults.py 

clean_FastResults_scirl.mat: clean
clean_FastResults_mwal.mat: clean
clean_SafeResults_scirl.mat: clean
clean_SafeResults_mwal.mat: clean

Comparison_conf.py: Comparison.org 
	$(call tangle,"Comparison.org")

Comparison_ErrorBars: Comparison_conf.py #FastResults_scirl.mat FastResults_mwal.mat SafeResults_scirl.mat SafeResults_mwal.mat
	mv Comparison_conf.py conf.py
	../../Utils/ErrorBars.py
	rm conf.py

FastResults_mwal.mat-0: Comparison_ErrorBars
FastResults_scirl.mat-0: Comparison_ErrorBars
SafeResults_mwal.mat-0: Comparison_ErrorBars
SafeResults_scirl.mat-0: Comparison_ErrorBars

FastResults_EB.gp: Comparison.org
	$(call tangle,"Comparison.org")

FastResults_EB.pdf: FastResults_EB.gp FastResults_mwal.mat-0 SafeResults_scirl.mat-0 FastResults_true.mat FastResults_random.mat
	gnuplot FastResults_EB.gp
	ps2pdf FastResults_EB.ps
	rm FastResults_EB.ps

SafeResults_EB.gp: Comparison.org
	$(call tangle,"Comparison.org")

SafeResults_EB.pdf: SafeResults_EB.gp SafeResults_mwal.mat-0 SafeResults_scirl.mat-0 SafeResults_true.mat SafeResults_random.mat
	gnuplot SafeResults_EB.gp
	ps2pdf SafeResults_EB.ps
	rm SafeResults_EB.ps

RandomResults.py: Comparison.org
	$(call tangle,"Comparison.org")

SafeResults_random.mat: al_code/safe_reward.mat al_code/fast_reward.mat
	$(call tangle,"Comparison.org")
	$(call tangle,"MWAL.org")
	$(call tangle,"Main.org")
	python RandomPlay.py  > PlayRandom.dat
	cat PlayRandom.dat | sed -e 's/\[1, 160]/\[1, 160, \[0, 0\]\]/' | awk '{if($$5) print $0}' | tr -d '[' | tr -d ']' | tr -d ',' | sed -e 's/left/0/' | sed -e 's/none/1/' | sed -e 's/right/2/' | sed -e 's/down/0/' | sed -e 's/up/2/' > PlayRandom_stripped.dat
	python tr2mu.py PlayRandom_stripped.dat | tr -d '[' | tr -d ']' > PlayRandom_mu.mat
	python RandomResults.py 

FastResults_random.mat: SafeResults_random.mat


App.py: Simulator.org
	$(call tangle,"Simulator.org")


RandomSamples.py: RandomSamples.org
	$(call tangle,"RandomSamples.org")

RandomSamples.dat: RandomSamples.py App.py
	python RandomSamples.py > toto
	cat toto | head -n $$(($$(cat toto | wc -l) -1 )) | tail -n $$(($$(cat toto | wc -l) -1 - 2 )) > tata
	cat tata | awk '{printf $$0" "($$4+$$5)*-1" ";if( $$9 == 10) print 0 ;else print 1}'> RandomSamples.dat
	rm toto tata

Highway_lspi.exe: Highway_lspi.o phipsi.o ../utils.o ../greedy.o ../LSTDQ.o ../LSPI.o 
	$(O2EXE) -o Highway_lspi.exe Highway_lspi.o phipsi.o ../utils.o ../greedy.o ../LSTDQ.o ../LSPI.o 

Highway_lspi.o: Highway_lspi.c ../utils.h ../LSPI.h ../greedy.h ../RL_Globals.h phipsi.h
	$(call c2obj,"Highway_lspi.c")

Highway_lspi.c: Main.org
	$(call tangle,"Main.org")

omega_lspi.mat: Highway_lspi.exe RandomSamples.dat actions.mat
	./Highway_lspi.exe > omega_lspi.mat

actions.mat:
	printf "0\n1\n2\n">actions.mat 

LAFEM_Exp6: omega_lspi.mat GreedyPlay.py
	python GreedyPlay.py

../utils.o:
	make -C .. utils.o

../greedy.o:
	make -C .. greedy.o

../LSTDQ.o:
	make -C .. LSTDQ.o

../abbeel2004apprenticeship.o:
	make -C .. abbeel2004apprenticeship.o

../LSTDmu.o:
	make -C .. LSTDmu.o

../criteria.o:
	make -C .. criteria.o

../LSPI.o:
	make -C .. LSPI.o

../utils.h:
	make -C .. utils.h

../greedy.h:
	make -C .. greedy.h

../LSTDQ.h:
	make -C .. LSTDQ.h

../abbeel2004apprenticeship.h:
	make -C .. abbeel2004apprenticeship.h

../LSTDmu.h:
	make -C .. LSTDmu.h

../criteria.h:
	make -C .. criteria.h

../LSPI.h:
	make -C .. LSPI.h

../RL_Globals.h:
	make -C .. RL_Globals.h

../IRL_Globals.h:
	make -C .. IRL_Globals.h

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs

O2EXE=gcc $(LFLAGS)

SCIRL_clean:
	find . -maxdepth 1 -iname "sa2saspsie.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlayFast_saspsie.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlaySafe_saspsie.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "SCIRL_LSTDmu.c"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "SCIRL_LSTDmu.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "SCIRL_LSTDmu.exe"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E_Fast_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "omega_mu_E_Safe_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "SCIRL_Exp6.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_Fast_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "theta_lafem_Safe_*.mat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "optimal_policy.m"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "make_F.m"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "make_THETA.m"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "scirl_fast_policy_*.txt"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "scirl_safe_policy_*.txt"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlayFast_scirl.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlaySafe_scirl.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlayFast_scirl_stripped.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlaySafe_scirl_stripped.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlayFast_scirl_mu.mat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "PlaySafe_scirl_mu.mat"   | xargs $(XARGS_OPT) rm


Simulator_clean:
	find . -maxdepth 1 -iname "App.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "App.pyc"   | xargs $(XARGS_OPT) rm

Comparison_clean:
	find . -maxdepth 1 -iname "ExpertsResults.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Comparison_conf.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "FastResults_*"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "SafeResults_*"   | xargs $(XARGS_OPT) rm

MWAL_clean:
	find . -maxdepth 1 -iname "PolicyPlay.py"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "fast_policy.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "fast_policy.txt"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "safe_policy.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "safe_policy.txt"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "fast_reward.mat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "safe_reward.mat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "my_MWAL.m"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "tr2mu.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlayFast.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlaySafe.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlayFast_*.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlayFast_*.mat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlaySafe_*.dat"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "PlaySafe_*.mat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "policy.dat"   | xargs $(XARGS_OPT) rm
	find al_code -maxdepth 1 -iname "policy_mwal_*"   | xargs $(XARGS_OPT) rm

Main_clean:
	find . -maxdepth 1 -iname "GreedyPlay.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "Highway_lspi.c"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.c"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.h"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.o"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.py"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "phipsi.pyc"   | xargs $(XARGS_OPT) rm

RandomSamples_clean:
	find . -maxdepth 1 -iname "RandomSamples.py"   | xargs $(XARGS_OPT) rm



clean: SCIRL_clean Comparison_clean Simulator_clean MWAL_clean Main_clean RandomSamples_clean
	find . -maxdepth 1 -iname "*~"   | xargs $(XARGS_OPT) rm

Makefile:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org

  CFLAGS+= -I..
c2obj=gcc $(CFLAGS) -c $1
