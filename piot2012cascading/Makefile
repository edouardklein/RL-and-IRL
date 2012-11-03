all:main.pdf

ECML13_cascading.pdf: ECML13_cascading.tex Biblio.bib #Fig1-NoX.pdf Fig2-ClassifAndUs.pdf Fig3-SCIRLAndUs.pdf Fig4-Legend.pdf
	pdflatex -interaction=batchmode ECML13_cascading.tex && bibtex ECML13_cascading && pdflatex -interaction=batchmode ECML13_cascading.tex && pdflatex -interaction=batchmode ECML13_cascading.tex 

RTEcompile:
	pdflatex -interaction=batchmode ECML13_cascading.tex

RTEstartView: RTEcompile
	xpdf -remote 'testRTE1' ECML13_cascading.pdf

RTEview:
	xpdf -remote 'testRTE1' -reload

RTEstopView:
	xpdf -remote 'testRTE1' -quit



Fig1-NoX.pdf: 4plots

Fig2-ClassifAndUs.pdf: 4plots

Fig3-SCIRLAndUs.pdf: 4plots

Fig4-Legend.pdf: 4plots

4plots: Cascading_Exp5.mat SCIRL_Exp3.mat 4plots.py
	python 4plots.py

