import os
import sys
import pyfits
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart

# Load the data.
f = pyfits.open(os.path.join(dirname, "data.fits"))
lc = np.array(f[1].data)
f.close()
time = lc["TIME"]
flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

# The limb-darkening parameters.
nbins, gamma1, gamma2 = 100, 0.39, 0.1
ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)

if True:
    ldp.bins = np.sqrt(np.linspace(0.0, 1.0, 15)[1:])
    rbins, ir = ldp.bins, ldp.intensity
    ir *= 1.0 / ir[0]
    ldp = bart.LimbDarkening(rbins, ir)

# Initialize the planetary system.
fs = np.median(flux)
iobs = 0.0
system = bart.BART(fs, iobs, ldp)

# The parameters of the planet:
r = 0.0247
a = 6.47
e = 0.0
T = 3.21346
phi = 2.90765788
i = 89.76

# Add the planet.
system.add_planet(r, a, e, T, phi, i)

# Fit it.
# print(system.optimize(time, flux, ferr, pars=[u"phi"]))
system.fit(time, flux, ferr, pars=[u"fs", u"phi", u"T", u"a", u"r",
                                   u"i", u"e", u"ldp"])
system.plot_fit()
system.plot_triangle()
