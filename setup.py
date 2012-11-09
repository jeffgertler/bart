#!/usr/bin/env python

import os
import sys

from numpy.distutils.core import setup, Extension


if "interface" in sys.argv:
    # Generate the Fortran signature/interface.
    cmd = "cd src;"
    cmd += "f2py lightcurve.f90 -m _bart -h ../bart/bart.pyf"
    cmd += " --overwrite-signature"
    os.system(cmd)
    sys.exit(0)

# Define the Fortran extension.
f_ext = Extension("bart._bart", ["bart/bart.pyf", "src/lightcurve.f90",
                                 "src/ld.f90", "src/orbit.f90"])

setup(
    name="bart",
    author="Dan Foreman-Mackey",
    packages=["bart"],
    ext_modules=[f_ext],
)
