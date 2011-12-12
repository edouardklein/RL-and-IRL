SUB_DIRS=ChainWalk GridWorld InvertedPendulum klein2011dimensionality

code:
	cat *.org > All.org; emacs -batch --visit All.org --funcall org-babel-tangle --script ~/.emacs; rm All.org &&\
	for dir in $(SUB_DIRS); do $(MAKE) -C $$dir code ; done 