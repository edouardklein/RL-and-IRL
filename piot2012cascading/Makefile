all:main.pdf

main.pdf: main_KLEIN_v3.tex Biblio.bib Fig1.pdf Fig2.pdf Fig3.pdf Legend.pdf
	pdflatex main_KLEIN_v3.tex && bibtex main_KLEIN_v3 && pdflatex main_KLEIN_v3.tex && pdflatex main_KLEIN_v3.tex
	mv main_KLEIN_v3.pdf main.pdf
	evince main.pdf

Fig1.pdf: Fig2.pdf

Fig2.pdf: ANIRL_Exp3.mat Cascading_Exp3.mat GridWorld_plot.py
	python GridWorld_plot.py

Fig3.pdf: Cascading_Exp5.mat Highway_plot.py
	python Highway_plot.py

Legend.pdf: Cascading_Exp5.mat Legend_plot.py
	python Legend_plot.py

