org2pdf=emacs -batch --visit $1.org --funcall org-export-as-latex --script ~/.emacs && pdflatex $1.tex && bibtex $1 && makeglossaries main && pdflatex $1.tex && pdflatex $1.tex

show: main.pdf
	open main.pdf

main.pdf: main.org
	$(call org2pdf,"main")

integration: non_interactive_spellcheck count_pages
	echo Spelling errors: $$(make -s non_interactive_spellcheck)
	echo $$(make -s count_pages)

count_pages: main.pdf
	pdflatex main.tex | egrep '[0-9]+ pages' | cut -d'(' -f2  | cut -d',' -f1

interactive_spellcheck: All.dict main.org
	hunspell -t -p ./All.dict main.org

non_interactive_spellcheck: All.dict main.org
	egrep -v '^\s*\#' main.org | hunspell -a -t -p ./All.dict | egrep -v '(^\*|^\s*$$|^\@)' | wc -l

All.dict: Citations.dict Glossary.dict Jargon.dict OrgMode.dict
	cat OrgMode.dict Citations.dict Glossary.dict Jargon.dict > All.dict

Citations.dict: main.org
	egrep  '\@.*\{' Biblio.bib | cut -d '{' -f2 | tr ',' '' > Citations.dict 

Glossary.dict: main.org
	egrep 'new(acronym|angl)\{' main.org | cut -d'{' -f2 | tr '}' '' > Glossary.dict

clean:
	rm -f *~
	rm -f main.aux main.tex main.bbl main.blg main.log main.out main.pdf main.toc main.acn main.acr main.alg  main.angl main.aot main.atn main.glg main.glo main.gls main.xdy
	rm -f Glossary.dict Citations.dict All.dict

fixme:
	egrep --color FIXME *
