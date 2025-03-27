# file: Makefile	George B. Moody (george@mit.edu)

PRODUCTS= paper.pdf

all: $(PRODUCTS)

paper.pdf:	paper.tex
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper
	rm -f paper.aux paper.blg paper.dvi paper.log
	rm -f paper.ps *~

paper.ps:	paper.dvi
	dvips -Ppdf -t letter -o paper.ps paper

paper.dvi:	paper.tex
	latex paper	# creates paper.aux, needed by bibtex
	bibtex paper	# creates paper.bbl, needed by latex
	latex paper	# merges references
	latex paper  # produces final copy with correct citation numbers

kit:
	make clean
	cd ..; tar cfvz latex.tar.gz latex
	cd ..; rm -f latex.zip; zip -r latex.zip latex

clean:
	# cd example1; make clean
	# cd example2; make clean
	rm -f template.aux template.bbl template.blg template.dvi template.log
	rm -f template.ps *~
	rm -f paper.aux paper.bbl paper.blg paper.dvi paper.log
	rm -f paper.ps *~
