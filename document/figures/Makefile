.PHONY: kepler6

all: geom.pdf kepler6

geom.pdf: geom.py utils.py
	python geom.py

kepler6:
	python kepler6.py -e 0.01 -e 0.05 -e 0.1 -e 0.5 kepler6/data/kplr010874614-2011271113734_slc.fits
	cp kepler6-0.05/ldp.png k6-ldp.png
	cp kepler6-0.05/ldp.pdf k6-ldp.pdf
	cp kepler6-0.05/corner.png k6-corner.png
	cp kepler6-0.05/corner.pdf k6-corner.pdf
	cp kepler6-0.05/lightcurves/0.png k6-lc.png
	cp kepler6-0.05/lightcurves/0.pdf k6-lc.pdf
	cp kepler6-0.05/table.tex kepler6.tex

# case1.pdf: cases.py utils.py
# 	python cases.py 1

clean:
	rm -f geom.pdf k6-*.png kepler6.tex
