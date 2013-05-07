
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
import bart
from bart import kepler
import numpy as np
import matplotlib.pyplot as pl

ldp = kepler.fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
star = bart.Star(mass=.602, radius=.9, ldp=ldp, flux = 100.)
planet = bart.Planet(a=162, r=11, mass=.0009551)
system = bart.PlanetarySystem(star, iobs=86.8)
system.add_planet(planet)
tbin = .001
t = np.arange(-200, 200, tbin)
pl.clf()
pl.plot(t, system.lightcurve(t))
pl.savefig("lightcurve_bug_test_r<1")

