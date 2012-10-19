all:main.pdf

ECML13_cascading.pdf: ECML13_cascading.tex Biblio.bib Fig3.pdf
	pdflatex -interaction=batchmode ECML13_cascading.tex && bibtex ECML13_cascading && pdflatex -interaction=batchmode ECML13_cascading.tex && pdflatex -interaction=batchmode ECML13_cascading.tex 

RTEcompile:
	pdflatex -interaction=batchmode ECML13_cascading.tex

RTEstartView: RTEcompile
	xpdf -remote 'testRTE1' ECML13_cascading.pdf

RTEview:
	xpdf -remote 'testRTE1' -reload

RTEstopView:
	xpdf -remote 'testRTE1' -quit



Fig1.pdf: Fig2.pdf

Fig2.pdf: ANIRL_Exp3.mat Cascading_Exp3.mat GridWorld_plot.py
	python GridWorld_plot.py

Fig3.pdf: Cascading_Exp5.mat Highway_plot.py
	python Highway_plot.py

Legend.pdf: Cascading_Exp5.mat Legend_plot.py
	python Legend_plot.py

