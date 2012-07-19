.PHONY: Makefile

SUB_DIRS=ChainWalk GridWorld InvertedPendulum

Makefile:
	cat *.org > All.org; tangle All.org ; rm All.org &&\
	for dir in $(SUB_DIRS); do $(MAKE) -C $$dir Makefile ; done 
