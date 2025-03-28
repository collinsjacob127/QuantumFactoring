[comment]: <> (file: README)

`make` builds [paper.pdf](paper.pdf) and [template.pdf](template.pdf). References are in [refs.bib](refs.bib).

This directory uses the CSCSU conference template for LaTeX, modified by Jacob Collins 3/26/2025.

Writing a CinC Paper Using LaTeX

George Moody (george@mit.edu)
Last revised: 20 February 2012
## Last update: Wed Oct  3 06:47:08 2018 by Rob
##    - Simplified and much updated template document/rules. 

This directory contains the CinC Author's Kit for preparing papers in
PDF format using LaTeX.  The following files are included:

 README		this file

 balance.sty	LaTeX macro for equalizing column length

 cinc.cls	CinC style file

 cinc.bst	CinC bibliography style file

 template.tex	generic template for a paper (LaTeX source)

 template.pdf	formatted version of template.tex

 refs.bib	sample BibTeX bibliography file

 Makefile	rules for generating template.pdf using a 'make' utility

 example	directory containing a sample paper, including
		figures and bibliography files

INSTALLATION

This kit can be used with any reasonably modern (September, 1994 or later)
version of LaTeX, a dialect of Donald Knuth's TeX software for typesetting.

If you don't already have LaTeX on your computer, download and install it now.
We recommend:

    TeX Live (http://www.tug.org/texlive/) for Linux or other Unix
    MacTeX (http://www.tug.org/mactex/) for Mac OS X
    proTeXt (http://www.tug.org/protext/) for MS Windows

All are free and include everything needed other than the CinC-specific files
included in this author's kit.

To install the files provided in this directory, do one of the following:

* Copy cinc.cls and cinc.bst to the directory that will contain your
  paper.

   -- or --

* Copy them to the standard locations for files of these types.  You may need
  to run texhash or mktexlsr afterwards, so that LaTeX and BibTeX can find
  these files.

If your installation of LaTeX doesn't include balance.sty, copy balance.sty
from this directory into the same location where you copied cinc.cls.

The instructions below assume that you will use pdflatex to process your
paper and to create a PDF file from it. The recommended TeX distributions
mentioned above, and most others, include pdflatex. If you don't use pdflatex,
use extra care to be sure that the page size and margins in your final PDF are
correct; see the Appendix for details.


FORMATTING THE EXAMPLES (OPTIONAL)

The use of these files is best illustrated by the included sample papers
(example1/example1.tex and example2/example2.tex), which are accompanied by
sets of PDF (.pdf), PostScript (.ps), and encapsulated PostScript (.eps)
figures in example*/figures/, and by BibTeX bibliography (.bib) files in
example*/bib/, all of which are referenced in the papers.  To format the sample
papers:

1. Install cinc.cls and cinc.bst by copying them from this directory into the
   example directories, or into the standard locations.

2. Enter the example1 directory and do the following:
	pdflatex example1   [creates example1.aux, needed by bibtex]
	bibtex example1	    [creates example1.bbl, needed by pdflatex]
	pdflatex example1   [merges references]
	pdflatex example1   [produces final PDF with correct citation numbers]

The multiple runs of pdflatex are needed to prepare data for bibtex and
then to resolve the cross-references. These steps produce example1.pdf (and
several temporary files that can be removed).

The 'Makefile' included in each directory can be used by a 'make' utility
to automate these steps.

A similar procedure can be used to format example2.tex, which illustrates
other features of the CinC style files.


WRITING YOUR PAPER

Make a copy of the generic template (template.tex). The instructions below
assume that you have saved your copy of the template as a file named paper.tex,
into which you can type your paper.

Please use the example files provided as models for your own papers. 

SUBMITTING YOUR PAPER

You must submit the LaTeX source for your paper together with any other files
needed to produce a copy of your paper, such as files containing figures,
tables, and references.  Collect them together in a zip file to upload to
the paper collection site.

Before submitting the zip file, test that it is complete and verify that the
files within it produce a properly formatted paper of no more than 4 pages.

You must submit your paper in PDF format (other formats, such as LaTeX
source, DVI, and PostScript, cannot be accepted).

You must submit your paper electronically (paper copies are no longer necessary
or accepted).  Instructions on submitting your paper using your web
browser are included in the CinC Author's Kit.


_______________________________________________________________________________

ACKNOWLEDGMENTS

The cinc.cls file was originally adapted by Hans Kestler from Peter Nuchter's
IEEEtran2e.cls, which in turn was adapted from IEEEtran.sty by Gerry Murray and
Silvano Balemi.  Bob Throne, Alan Murray, Andrew Sims, George Moody, and Erik
Bojorges contributed feedback, bug fixes, and further improvements.  The
cinc.bst file was created by Hans Kestler and George Moody using Patrick Daly's
makebst generator.  George Moody wrote the generic template.

Your comments, suggestions, questions, and bug reports are welcome;  please
send them to Rob MacLeod (macleod@sci.utah.edu).

_______________________________________________________________________________
