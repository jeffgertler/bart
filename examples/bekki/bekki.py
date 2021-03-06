#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import os
import sys
import emcee
import time as timer
import triangle
import matplotlib.pyplot as pl
import numpy as np
import kplr
from kplr.ld import get_quad_coeffs
import untrendy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart
from bart.parameters import Parameter, ImpactParameter, PeriodParameter
from bart.priors import UniformPrior

client = kplr.API()


def setup(gp=True):
    client = kplr.API()

    # Query the KIC and get some parameters.
    kic = client.star(7364176)
    teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
    assert teff is not None

    # Get the limb darkening law.
    mu1, mu2 = get_quad_coeffs(teff, logg=logg, feh=feh)
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

    # Build the star object.
    star = bart.Star(ldp=ldp)

    # Set up the planet.
    prng = 5.0
    period = 272.1884457597957407
    size = 0.01
    epoch = 103.7235285266694973 + 0.265
    a = star.get_semimajor(period)
    b = 0.1
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
            tf = time_[m]
            fl = flux_[m]
            fle = ferr_[m]

            if not gp:
                mu = untrendy.median(tf, fl, dt=4.0)
                fl /= mu
                fle /= mu

            model.datasets.append(dsc(tf, fl, fle, alpha=1.0, l2=3.0,
                                      dtbin=None))

    # Add some priors.
    dper = prng / (maxn - minn)

    # Add some parameters.
    model.parameters.append(Parameter(planet, "t0"))
    model.parameters.append(Parameter(planet, "r"))
    model.parameters.append(ImpactParameter(planet))

    # Prior range for the period so that it doesn't predict transits outside
    # of the data range.
    ppr = UniformPrior(period - dper, period + dper)
    model.parameters.append(PeriodParameter(planet, lnprior=ppr))

    return model, period


if __name__ == "__main__":
    import sys

    fn = "samples.txt"
    model, period = setup()

    if "--no-data" not in sys.argv:
        [pl.plot(d.time % period, d.flux + 0.003 * i, ".", ms=2)
         for i, d in enumerate(model.datasets)]
        pl.savefig("data.png")

    if "--results" not in sys.argv:
        # Set up sampler.
        nwalkers = 20
        v = model.vector
        p0 = v[None, :] * (1e-4 * np.random.randn(len(v) * nwalkers)
                           + 1).reshape((nwalkers, len(v)))
        sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), model,
                                        threads=nwalkers)

        # Run a burn-in.
        pos, lnprob, state = sampler.run_mcmc(p0, 100)
        sampler.reset()

        with open(fn, "w") as f:
            f.write("# {0}\n".format(" ".join(map(unicode, model.parameters))))

        strt = timer.time()
        for pos, lnprob, state in sampler.sample(pos, lnprob0=lnprob,
                                                 iterations=1000,
                                                 storechain=False):
            with open(fn, "a") as f:
                for p, lp in zip(pos, lnprob):
                    f.write("{0} {1}\n".format(
                        " ".join(map("{0}".format, p)), lp))

        print("Took {0} seconds".format(timer.time() - strt))
        print("Acceptance fraction: {0}"
              .format(np.mean(sampler.acceptance_fraction)))

    samples = np.loadtxt(fn)
    figure = triangle.corner(samples)
    figure.savefig("triangle.png")

    P = np.median(samples[:, -2])
    print(period, P)

    ax = pl.figure().add_subplot(111)
    [ax.plot(d.time % P, d.flux + 0.002 * i, ".", ms=2)
     for i, d in enumerate(model.datasets)]

    mn, mx = ax.get_xlim()
    for sample in samples[::831]:
        model.vector = sample[:1]
        [ax.plot(d.time % P, d.predict(model) + 0.002 * i, "k",
                 alpha=0.05)
         for i, d in enumerate(model.datasets)]

    pl.savefig("samples.png")
