#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import numpy as np
import matplotlib.pyplot as pl
import kplr
from kplr.ld import get_quad_coeffs
import untrendy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart

client = kplr.API()


def setup(gp=True):
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
    period = 278.
    size = 0.03
    epoch = 20.0
    a = star.get_semimajor(period)
    b = 0.3
    incl = np.degrees(np.arctan2(a, b))
    planet = bart.Planet(size, a, t0=epoch)

    # Set up the system.
    ps = bart.PlanetarySystem(star, iobs=incl)
    ps.add_planet(planet)

    # Load the data and inject into each transit.
    lcs = kic.get_light_curves(short_cadence=False, fetch=False)
    datasets = []
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

        # Inject the transit.
        flux_ *= ps.lightcurve(time_)
        mu = np.median(flux_)

        datasets.append(bart.data.LightCurve(time_, flux_ / mu, ferr_ / mu))

    return datasets, period


if __name__ == "__main__":
    datasets, period = setup()
    periods = np.arange(period - 1, period + 1, 0.2)

    lnlike = bart.search(periods, 0.03, datasets, alpha=1.0)

    pl.plot(periods, lnlike)
    pl.gca().axvline(period)
    pl.savefig("test.png")
