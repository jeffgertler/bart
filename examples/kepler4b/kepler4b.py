import pyfits
import numpy as np
import matplotlib.pyplot as pl

import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart

# Load the data.
f = pyfits.open(os.path.join(dirname, "data.fits"))
lc = np.array(f[1].data)
f.close()

time = lc["TIME"]
flux_raw = lc["SAP_FLUX"]
flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

t0 = int(np.median(time[~np.isnan(time)]))
time = time - t0

# Plot the raw data.
ax = pl.axes([0.15, 0.15, 0.8, 0.8])
ax.plot(time,
        1000 * (flux_raw / np.median(flux_raw[~np.isnan(flux_raw)]) - 1),
        ".k", alpha=0.5)
ax.set_xlabel("Time [days]")
ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")
pl.savefig("kepler4b_raw.png", dpi=300)

# The limb-darkening parameters.
nbins, gamma1, gamma2 = 100, 0.39, 0.1
ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)

if True:
    # Fit for the LDP with better spacing.
    ldp.bins = np.sqrt(np.linspace(0.0, 1.0, 15)[1:])
    rbins, ir = ldp.bins, ldp.intensity
    ir *= 1.0 / ir[0]
    ldp = bart.LimbDarkening(rbins, ir)

# Initialize the planetary system.
fstar = 1.0
mstar = 1.0
iobs = 0.0

# The parameters of the planet:
r = 0.0247
a = 6.47
e = 0.0
T = 3.21346
t0 = 0.4
pomega = 0.0
i = 89.76

rstar = bart.get_mstar(a, T)

# Add the planet.
system = bart.BART(fstar, mstar, rstar, iobs, ldp)
system.add_planet(r, a, e, t0, pomega, i)

# Fit it.
system.fit(time, flux, ferr, pars=[u"fs", u"phi", u"T", u"a", u"r", u"ldp"],
                        niter=5000, thin=100, ntrim=1, nwalkers=64)
system.plot_fit()
system.plot_triangle()
