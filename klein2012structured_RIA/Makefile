all: scirl_ria.pdf 

scirl_ria.pdf : scirl_ria.tex
	pdflatex scirl_ria.tex 
	bibtex scirl_ria
	pdflatex scirl_ria.tex 
	pdflatex scirl_ria.tex 
	evince scirl_ria.pdf

clean:
	find . -maxdepth 1 -iname "scirl_nips.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.log"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.out"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.aux"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.bbl"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.blg"   | xargs $(XARGS_OPT) rm
