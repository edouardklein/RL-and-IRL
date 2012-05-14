all:main.pdf

main.pdf: main_KLEIN_v3.tex Biblio.bib Fig1.pdf Fig2.pdf
	pdflatex main_KLEIN_v3.tex && bibtex main_KLEIN_v3 && pdflatex main_KLEIN_v3.tex && pdflatex main_KLEIN_v3.tex
	mv main_KLEIN_v3.pdf main.pdf
	evince main.pdf

Fig1.pdf: Fig2.pdf

Fig2.pdf: ANIRL_Exp3.mat Cascading_Exp3.mat GridWorld_plot.py
	python GridWorld_plot.py

