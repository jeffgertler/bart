LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})

TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out
SUFF        = pdf

CHECK_RERUN = grep Rerun $*.log

NAME    = doc
DOC_OUT = ${NAME}.${SUFF}

all: figures ${DOC_OUT}

%.pdf: %.tex figures/*.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."

figures: force_look
	cd figures;${MAKE} ${MFLAGS}

force_look:
	true

clean:
	${RM_TMP}
	cd figures;${MAKE} clean
