F90 = gfortran
FFLAGS = -O0

MANDEL_SRC = ../lib/transit/occultquad.f

gen_test_data: gen_test_data.f90
	${F90} ${FFLAGS} gen_test_data.f90 ${MANDEL_SRC} -o gen_test_data

clean:
	rm -rf gen_test_data
