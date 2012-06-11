all:main.pdf

tangle=emacs -batch --visit $1 --funcall org-babel-tangle --script ~/.emacs >/dev/null 2>/dev/null

main.tex: main.org 
	$(call tangle,"main.org")


main.pdf: main.tex Biblio.bib Fig1.pdf Fig2.pdf Fig3.pdf Legend.pdf
	pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
	evince main.pdf

Fig1.pdf: Fig2.pdf

Fig2.pdf: ANIRL_Exp3.mat Cascading_Exp3.mat GridWorld_plot.py
	python GridWorld_plot.py

Fig3.pdf: Cascading_Exp5.mat Highway_plot.py
	python Highway_plot.py

Legend.pdf: Cascading_Exp5.mat Legend_plot.py
	python Legend_plot.py

