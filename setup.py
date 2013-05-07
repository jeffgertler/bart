#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import sys

from numpy.distutils.core import setup, Extension


if sys.argv[-1] == "publish":
    os.system("git rev-parse --short HEAD > COMMIT")
    os.system("python setup.py sdist upload")
    sys.exit()


# First, make sure that the f2py interfaces exist.
interface_exists = os.path.exists("bart/bart.pyf")
if "interface" in sys.argv or not interface_exists:
    # Generate the Fortran signature/interface.
    cmd = "cd src;"
    cmd += "f2py lightcurve.f90 orbit.f90 ld.f90 discontinuities.f90"
    cmd += " -m _bart -h ../bart/bart.pyf"
    cmd += " --overwrite-signature"
    os.system(cmd)

    if "interface" in sys.argv:
        sys.exit(0)

# Define the Fortran extension.
bart = Extension("bart._bart", ["bart/bart.pyf", "src/lightcurve.f90",
                                "src/ld.f90", "src/orbit.f90",
                                "src/discontinuities.f90"])

# Get version.
vre = re.compile("__version__ = \"(.*?)\"")
m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "bart", "__init__.py")).read()
version = vre.findall(m)[0]

# Get the installation requirements.
install_requires = []
for l in open("requirements.txt"):
    if "#" in l:
        l = l[:l.index("#")]
    l = l.strip()
    if len(l) == 0:
        continue
    install_requires.append(l)

setup(
    name="bart",
    url="http://dan.iel.fm/bart",
    version=version,
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="Rapid Exoplanet Transit Modeling in Python",
    long_description=open("README.rst").read(),
    packages=["bart", "bart.parameters"],
    package_data={"": ["README.rst"], "bart": ["ldcoeffs/sing09.txt"]},
    package_dir={"bart": "bart"},
    include_package_data=True,
    ext_modules=[bart],
    install_requires=install_requires,
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
