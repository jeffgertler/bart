import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
import bart


# The un-occulted flux of the star.
fs = 1e4

# The oservation angle of the planetary disk in degrees (zero is edge-on
# just to be difficult).
iobs = 0.0

# The limb-darkening parameters.
ld_type = "quad"
nbins, gamma1, gamma2 = 10, 0.39, 0.1
ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)
true_ldp = ldp.plot()

# Initialize the planetary system.
system = bart.BART(fs, iobs, ldp=ldp)

# The parameters of the planet:
r = 0.2      # The radius of the planet in fractional radii.
a = 6.0      # The semi-major axis of the orbit in fractional radii.
e = 0.01     # The eccentricity of the orbit.
T = 4.2      # The period of the orbit in days.
phi = np.pi  # The phase of the orbit in radians.
i = 0.1      # The relative observation angle for this planet in degrees.

# Add the planet.
system.add_planet(r, a, e, T, phi, i)

# Compute some synthetic data.
time = 365.0 * np.random.rand(1000)
ferr = 10 * np.random.rand(len(time))  # The uncertainties.
flux = system.lightcurve(time) + ferr * np.random.randn(len(time))

# Decrease the number of bins in LDP.
# ldp.bins = np.log(np.linspace(0.0, 1.0, 15)[1:]) / 5.0 + 1
# ldp.gamma1, ldp.gamma2 = 0.5, 0.2
# rbins, ir = ldp.bins, ldp.intensity
# ir *= 1.0 / ir[0]
# system.ldp = bart.LimbDarkening(rbins, ir)

# Fit it.
chain = system.fit(time, flux, ferr,
                   pars=["fs", "T", "a", "r", "ldp"])

system.plot_triangle(truths={"fs": fs, "T0": T, "a0": a, "r0": r})
system.plot_fit(true_ldp=true_ldp)
