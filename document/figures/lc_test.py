
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

np.random.seed(100)

'''
import matplotlib.mlab as mlab
delta = .025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
print(X)
print(Y)
print(Z1)
'''


ldp = kepler.fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
star = bart.Star(mass=.602, radius=1, ldp=ldp, flux = 100.)
planet = bart.Planet(a=130, r=21, mass=.0009551)
system = bart.PlanetarySystem(star, iobs=86.8)
system.add_planet(planet)

tbin = .001
t = np.arange(-100, 100, tbin)
system.get_photons(t)

#set the parameter to check over
brightness_test = (np.arange(0, 10)+.5) *20
angle_test = np.arange(0, 7, .7) +.5
X, Y = np.meshgrid(brightness_test, angle_test)

Z = np.empty_like(X)
for i in range(len(angle_test)):
    print("angle test", angle_test[i])
    planet.ix=angle_test[i]
    for j in range(len(brightness_test)):
        star.flux = brightness_test[j]
        system.get_photons(t, plot=False)
        eclipse_prob = system.poissonlike()
        r = planet.r
        planet.r = 0
        constant_prob = system.poissonlike()
        planet.r = r
        Z[i][j] = eclipse_prob-constant_prob
        print(Z[i][j])

print(Z)
pl.clf()
plot = pl.contour(X, Y, Z)
pl.clabel(plot, inline=1, fontsize=10)




#pl.plot(param_test, eclipse_prob, 'r', label='eclipse')
#pl.plot(param_test, constant_prob, 'k', label='constant')
#pl.legend(('eclipse', 'constant'), loc='lower right')
#pl.ylim(0, np.max(eclipse_prob)*1.05)
pl.xlabel("brighness of star (photons/sec)")
pl.ylabel("angle of orbital plane to observation")

pl.savefig("brightness_test")



