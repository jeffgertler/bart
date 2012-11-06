FIGURES = geom.pdf case1.pdf

all: $(FIGURES)

geom.pdf: geom.py utils.py
	python geom.py

case1.pdf: cases.py utils.py
	python cases.py 1

clean:
	rm -f $(FIGURES)
