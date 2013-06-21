#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import emcee
import triangle
import matplotlib.pyplot as pl
import numpy as np
import kplr
from kplr.ld import get_quad_coeffs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart

client = kplr.API()


def setup():
    client = kplr.API()

    # Query the KIC and get some parameters.
    # kic = client.star(8415109)  # Bright variable.
    kic = client.star(2301306)  # Quiet G-type.
    teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
    assert teff is not None

    # Get the limb darkening law.
    mu1, mu2 = get_quad_coeffs(teff, logg=logg, feh=feh)
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

    # Build the star object.
    star = bart.Star(ldp=ldp)

    # Set up the planet.
    prng = 5
    period = 278.
    size = 0.03
    epoch = 10.0
    a = star.get_semimajor(period)
    b = 0.3
    incl = np.degrees(np.arctan2(a, b))
    planet = bart.Planet(size, a, t0=epoch)

    # Set up the system.
    ps = bart.PlanetarySystem(star, iobs=incl)
    ps.add_planet(planet)

    # Initialize the model.
    model = bart.Model(ps)

    # Load the data and inject into each transit.
    lcs = kic.get_light_curves(short_cadence=False, fetch=False)

    # Loop over the datasets and read in the data.
    minn, maxn = 1e10, 0
    for lc in lcs:
        with lc.open() as f:
            # The light curve data are in the first FITS HDU.
            hdu_data = f[1].data
            time_, flux_, ferr_, quality = [hdu_data[k]
                                            for k in ["time", "sap_flux",
                                                      "sap_flux_err",
                                                      "sap_quality"]]

        # Mask the missing data.
        mask = (np.isfinite(time_) * np.isfinite(flux_) * np.isfinite(ferr_)
                * (quality == 0))
        time_, flux_, ferr_ = [v[mask] for v in [time_, flux_, ferr_]]

        # Cut out data near transits.
        hp = 0.5 * period
        inds = np.abs((time_ - epoch + hp) % period - hp) < prng
        if not np.sum(inds):
            continue
        time_, flux_, ferr_ = [v[inds] for v in [time_, flux_, ferr_]]

        # Inject the transit.
        flux_ *= ps.lightcurve(time_)

        tn = np.array(np.round(np.abs((time_ - epoch) / period)), dtype=int)
        alltn = set(tn)

        maxn = max([maxn, max(alltn)])
        minn = min([minn, min(alltn)])

        for n in alltn:
            m = tn == n
            model.datasets.append(bart.data.LightCurve(time_[m], flux_[m],
                                                       ferr_[m]))

    # Add some priors.
    dper = prng / (maxn - minn)
    model.lnpriors.append(lambda m: 0.0 if period - dper <
                          m.planetary_system.planets[0].get_period(1.0)
                          < period + dper else -np.inf)

    # Add some parameters.
    model.parameters.append(bart.parameters.PlanetParameter("t0"))
    model.parameters.append(bart.parameters.PlanetParameter("r"))
    model.parameters.append(bart.parameters.ImpactParameter())
    model.parameters.append(bart.parameters.PeriodParameter())

    return model, period


if __name__ == "__main__":
    model, period = setup()
    print(model.lnprob())

    [pl.plot(d.time % period, d.flux + 0.01 * i, ".")
     for i, d in enumerate(model.datasets)]
    pl.savefig("data.png")

    # Set up sampler.
    nwalkers = 20
    v = model.vector
    p0 = v[None, :] * (1e-4 * np.random.randn(len(v) * nwalkers)
                       + 1).reshape((nwalkers, len(v)))
    sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), model,
                                    threads=nwalkers)

    # Run a burn-in.
    pos, lnprob, state = sampler.run_mcmc(p0, 50)
    sampler.reset()

    fn = "samples.txt"
    with open(fn, "w") as f:
        f.write("# t0 r/R b/R P ln(prob)\n")
    for pos, lnprob, state in sampler.sample(pos, lnprob0=lnprob,
                                             iterations=100,
                                             storechain=False):
        with open(fn, "a") as f:
            for p, lp in zip(pos, lnprob):
                f.write("{0} {1}\n".format(
                    " ".join(map("{0}".format, p)), lp))

    print("Acceptance fraction: {0}"
          .format(np.mean(sampler.acceptance_fraction)))

    samples = np.loadtxt(fn)
    figure = triangle.corner(samples)
    figure.savefig("triangle.png")
