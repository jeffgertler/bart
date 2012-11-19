import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
import bart


# The radius in Solar radii and the un-occulted flux of the star.
rs, fs = 1.5, 1000.0

# The oservation angle of the planetary disk in degrees (zero is edge-on
# just to be difficult).
iobs = 0.0

# The limb-darkening parameters.
ld_type = "quad"
nbins, gamma1, gamma2 = 50, 0.39, 0.1

# Initialize the planetary system.
system = bart.BART(rs, fs, iobs,
                   ldp=bart.QuadraticLimbDarkening(nbins, gamma1, gamma2))

# The parameters of the planet:
r = 5.0      # The radius of the planet in Jupter radii.
a = 0.05     # The semi-major axis of the orbit in AU.
e = 0.01     # The eccentricity of the orbit.
T = 4.2      # The period of the orbit in days.
phi = np.pi  # The phase of the orbit in radians.
i = 0.1      # The relative observation angle for this planet in degrees.

# Add the planet.
system.add_planet(r, a, e, T, phi, i)

# Compute some synthetic data.
time = 365.0 * np.random.rand(100)
ferr = 50 * np.random.rand(len(time))  # The uncertainties.
flux = system.lightcurve(time) + ferr * np.random.randn(len(time))

# import matplotlib.pyplot as pl
# pl.plot(time % T, flux, ".k")
# pl.savefig("data.png")
# assert 0

# Fit it.
chain = system.fit(time, flux, ferr,
                   pars=["fs", "T", "a", "r", "gamma1", "gamma2"])

system.plot_fit()
