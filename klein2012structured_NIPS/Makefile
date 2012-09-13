all: scirl_nips_cr.pdf 

scirl_nips_cr.pdf : scirl_nips_cr.tex
	pdflatex scirl_nips_cr.tex 
	bibtex scirl_nips_cr
	pdflatex scirl_nips_cr.tex 
	pdflatex scirl_nips_cr.tex 
clean:
	rm  scirl_nips_cr.pdf *.log *.out *.aux *.bbl *.blg
