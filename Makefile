all:
	pdflatex main.tex && pdflatex main.tex

bib:
	pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

clean:
	rm *.aux *.bbl *.blg *.lof  *.log *.out *.toc *~
