#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
except ImportError:
    get_numpy_include_dirs = lambda: []

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

bart = Extension("bart._bart", ["bart/_bart.c", "src/bart.c", "src/kepler.c"],
                 include_dirs=["include"] + get_numpy_include_dirs())
george = Extension("bart._george", ["bart/_george.c", "src/george.cpp",
                                    "src/kernels.c"],
                   include_dirs=["include"] + get_numpy_include_dirs())
turnstile = Extension("bart._turnstile", ["bart/_turnstile.c",
                                          "src/turnstile.cpp",
                                          "src/kernels.c",
                                          "src/bart.c"],
                      include_dirs=["include"] + get_numpy_include_dirs())

setup(
    name="bart",
    url="http://dan.iel.fm/bart",
    version="0.1.0",
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="Rapid Exoplanet Transit Modeling in Python",
    long_description=open("README.rst").read(),
    packages=["bart"],
    package_data={"": ["README.rst"]},
    package_dir={"bart": "bart"},
    include_package_data=True,
    ext_modules=[bart, george, turnstile],
    install_requires=[
        "numpy",
    ],
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
