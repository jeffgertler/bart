#!/usr/bin/env python

import re
import os
import sys

from numpy.distutils.core import setup, Extension


if sys.argv[-1] == "publish":
    os.system("git rev-parse --short HEAD > COMMIT")
    os.system("python setup.py sdist upload")
    sys.exit()


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

# Get version.
vre = re.compile("__version__ = \"(.*?)\"")
m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "bart", "__init__.py")).read()
version = vre.findall(m)[0]

setup(
    name="bart",
    version=version,
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst"]},
    include_package_data=True,
    packages=["bart", ],
    ext_modules=[bart, kmeans, period],
    install_requires=["emcee", "triangle_plot"],
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
