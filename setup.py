#!/usr/bin/env python

import os
import sys

from numpy.distutils.core import setup, Extension


# First, make sure that the f2py interfaces exist.
interfaces_exist = [os.path.exists(p) for p in [u"bart/bart.pyf",
                                                u"bart/period/period.pyf"]]


if u"interface" in sys.argv or not all(interfaces_exist):
    # Generate the Fortran signature/interface.
    cmd = u"cd src;"
    cmd += u"f2py lightcurve.f90 orbit.f90 ld.f90 -m _bart -h ../bart/bart.pyf"
    cmd += u" --overwrite-signature"
    os.system(cmd)

    # And the same for the periodogram interface.
    cmd = u"cd src/period;"
    cmd += u"f2py periodogram.f90 -m _period -h ../../bart/period/period.pyf"
    cmd += u" --overwrite-signature"
    os.system(cmd)

    if u"interface" in sys.argv:
        sys.exit(0)

# Define the Fortran extension.
bart = Extension("bart._bart", ["bart/bart.pyf", "src/lightcurve.f90",
                                "src/ld.f90", "src/orbit.f90"])

# Define the K-means C extension.
kmeans = Extension("bart._algorithms", ["bart/_algorithms.c", ])

# Define the periodogram extension.
period = Extension("bart.period._period", ["bart/period/period.pyf",
                                           "src/period/periodogram.f90"])

setup(
    name=u"bart",
    author=u"Dan Foreman-Mackey",
    packages=[u"bart"],
    ext_modules=[bart, kmeans, period],
)
