F90 = gfortran
FFLAGS = -O0

MANDEL_SRC = lib/transit/occultquad.f
OUR_SRC = src/ld.f90

benchmark: src/benchmark.f90 $(MANDEL_SRC) $(OUR_SRC)
	$(F90) $(FFLAGS) -o bin/benchmark src/benchmark.f90 $(MANDEL_SRC) $(OUR_SRC)
