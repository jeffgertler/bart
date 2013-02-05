#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

__all__ = ["Star", "Planet", "PlanetarySystem", "LimbDarkening",
           "QuadraticLimbDarkening"]

from .bart import Star, Planet, PlanetarySystem
from .ldp import LimbDarkening, QuadraticLimbDarkening

__version__ = "0.0.2"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Dan Foreman-Mackey"
__contributors__ = [
                    "David W. Hogg @davidwhogg",
                    "Patrick Cooper @patrickcooper",
                   ]

# Get the git commit hash and save it as the __commit__ variable.
import os
import subprocess as sp
bart_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
commit_file = os.path.join(bart_dir, "COMMIT")
p = sp.Popen(["git", "rev-parse", "--short", "HEAD"],
             stdout=sp.PIPE, cwd=bart_dir)
r = p.wait()
__commit__ = None
if r == 0:
    __commit__ = p.stdout.read().strip()
elif os.path.exists(commit_file):
    __commit__ = open(commit_file).read().strip()
