default:
	pdflatex Cascading.tex
	bibtex Cascading
	pdflatex Cascading.tex
	pdflatex Cascading.tex

slides.pdf:slides.org
	emacs -batch --visit slides.org --funcall org-export-as-latex --script ~/.emacs
	pdflatex slides.tex
	pdflatex slides.tex


clean:
	rm -f *~ *.aux *.blg *.bbl *.log
