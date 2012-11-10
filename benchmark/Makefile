F90 = gfortran
FFLAGS = -O0

MANDEL_SRC = ../lib/transit/occultquad.f
OUR_SRC = ../src/ld.f90

default: scalings.pdf

run: benchmark.f90 $(MANDEL_SRC) $(OUR_SRC)
	$(F90) $(FFLAGS) -o run benchmark.f90 $(MANDEL_SRC) $(OUR_SRC)

bench.dat: run
	@echo "Running the benchmark... this might take a while."
	./run > bench.dat

scalings.pdf: bench.dat plot_benchmark.py
	python plot_benchmark.py bench.dat

clean:
	rm -f run bench.dat scalings.pdf
