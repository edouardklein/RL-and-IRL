all: scirl_nips_cr.pdf 

scirl_nips_cr.pdf : scirl_nips_cr.tex
	pdflatex scirl_nips_cr.tex 
	bibtex scirl_nips_cr
	pdflatex scirl_nips_cr.tex 
	pdflatex scirl_nips_cr.tex 
clean:
	find . -maxdepth 1 -iname "scirl_nips_cr.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.log"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.out"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.aux"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.bbl"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.blg"   | xargs $(XARGS_OPT) rm
