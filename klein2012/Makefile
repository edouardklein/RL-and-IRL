main.pdf: main.tex Biblio.bib
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	find . -maxdepth 1 -iname "headertikz.aux"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "slides.tex"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "slides.pdf"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.log"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.out"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.aux"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.bbl"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.blg"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.nav"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.snm"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*~"   | xargs $(XARGS_OPT) rm
	find . -maxdepth 1 -iname "*.toc"   | xargs $(XARGS_OPT) rm

