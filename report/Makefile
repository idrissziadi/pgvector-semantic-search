.PHONY: pdf clean

pdf:
	pdflatex rapport.tex
	biber rapport
	pdflatex rapport.tex
	pdflatex rapport.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.bcf *.run.xml
