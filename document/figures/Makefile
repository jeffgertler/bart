.PHONY: kepler6

all: geom.pdf kepler6

geom.pdf: geom.py utils.py
	python geom.py

kepler6:
	cp kepler6/ldp.png k6-ldp.png
	cp kepler6/corner.png k6-corner.png
	cp kepler6/lightcurves/0.png k6-lc.png
	cp kepler6/table.tex kepler6.tex

# case1.pdf: cases.py utils.py
# 	python cases.py 1

clean:
	rm -f geom.pdf k6-*.png kepler6.tex
