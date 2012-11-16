import os
import sys
import numpy as np
import matplotlib.pyplot as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
import bart


# The radius in Solar radii and the un-occulted flux of the star.
rs, fs = 1.5, 1000.0

# The oservation angle of the planetary disk in degrees (zero is edge-on
# just to be difficult).
iobs = 0.01

# The limb-darkening parameters.
ld_type = "quad"
nbins, gamma1, gamma2 = 50, 0.39, 0.1

# Initialize the planetary system.
system = bart.BART(rs, fs, iobs, ldp=[nbins, gamma1, gamma2], ldptype=ld_type)

# The parameters of the planet:
r = 5.0      # The radius of the planet in Jupter radii.
a = 0.05     # The semi-major axis of the orbit in AU.
e = 0.01     # The eccentricity of the orbit.
T = 4.2      # The period of the orbit in days.
phi = np.pi  # The phase of the orbit in radians.
i = 0.0      # The relative observation angle for this planet in degrees.

# Add the planet.
system.add_planet(r, a, e, T, phi, i)

# Compute a lightcurve.
time = 365.0 * np.random.rand(500)
ferr = 10 * np.random.rand(len(time))  # The uncertainties.
flux = system.lightcurve(time) + ferr * np.random.randn(len(time))

# Plot it.
pl.figure(figsize=(6, 8))

pl.subplot(3, 1, 1)
pl.errorbar(time, flux, yerr=ferr, fmt=".k")

pl.subplot(3, 1, 2)
pl.errorbar(time % T, flux, yerr=ferr, fmt=".k")
t = np.linspace(0, T, 1000)
f = system.lightcurve(t)
pl.plot(t, f, "#4682b4", lw=1)

pl.subplot(3, 1, 3)
pl.errorbar(time % T, flux, yerr=ferr, fmt=".k")
pl.plot(t, f, "#4682b4", lw=1)
pl.xlim(1.5, 2.7)

pl.savefig("lc.png")
