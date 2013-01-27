#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from .bart import Star, Planet, PlanetarySystem
from .ldp import LimbDarkening, QuadraticLimbDarkening, NonlinearLimbDarkening
from .parameters import (Parameter, LogParameter, MultipleParameter,
                         EccentricityParameters)

__version__ = "0.0.2"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Dan Foreman-Mackey"
__contributors__ = [
                    "David W. Hogg @davidwhogg",
                    "Patrick Cooper @patrickcooper",
                   ]
