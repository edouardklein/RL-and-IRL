all: scirl_ria.pdf 

scirl_ria.pdf : scirl_ria.tex
	pdflatex scirl_ria.tex 
	bibtex scirl_ria
	pdflatex scirl_ria.tex 
	pdflatex scirl_ria.tex 
	evince scirl_ria.pdf

clean:
	rm  scirl_ria.pdf *.log *.out *.aux *.bbl *.blg
