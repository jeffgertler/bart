F90 = gfortran

SRC_DIR = src
FILES = ${SRC_DIR}/orbit.f90 ${SRC_DIR}/ld.f90 ${SRC_DIR}/lightcurve.f90

default: bart

bart: ${FILES} ${SRC_DIR}/bart.f90
	mkdir -p bin
	${F90} ${FILES} ${SRC_DIR}/bart.f90 -o bin/bart
