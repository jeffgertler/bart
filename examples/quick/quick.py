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
nbins, gamma1, gamma2 = 100, 0.39, 0.1
ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)
true_ldp = ldp.plot()
print len(true_ldp[0]), len(true_ldp[1])

# Initialize the planetary system.
system = bart.BART(rs, fs, iobs, ldp=ldp)

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
time = 365.0 * np.random.rand(1000)
ferr = 5 * np.random.rand(len(time))  # The uncertainties.
flux = system.lightcurve(time) + ferr * np.random.randn(len(time))

# Decrease the number of bins in LDP.
ldp.bins = (np.linspace(0.0, 1.0, 11) ** 0.3)[1:]
ldp.gamma1, ldp.gamma2 = 0.5, 0.2
rbins, ir = ldp.bins, ldp.intensity
ir = 1.0 / (1 + rbins)
ir[0] = 1.0
system.ldp = bart.LimbDarkening(rbins, ir)

# Fit it.
chain = system.fit(time, flux, ferr,
                   pars=["fs", "T", "a", "r", "ldp"])

system.plot_fit(truths={"fs": fs, "T0": T, "a0": a, "r0": r},
                true_ldp=true_ldp)
