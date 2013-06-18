#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import sys
import numpy as np
import matplotlib.pyplot as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart
import kplr
from kplr.ld import get_quad_coeffs
import untrendy

client = kplr.API()


def inject(kicid):
    # Get the KIC entry.
    kic = client.star(kicid)
    teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
    radius = kic.kic_radius
    assert teff is not None

    # Compute the stellar parameters.
    if radius is None:
        radius = 1.0
    if logg is None:
        mass = 1.0
    else:
        mass = 1.0
        # g = 10 ** (logg - 2) / 6.955e8
        # mass = g * radius * radius / _G

    # Get the limb darkening law.
    mu1, mu2 = get_quad_coeffs(teff=teff, logg=logg, feh=feh, model="claret11")
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

    # Build the star object.
    star = bart.Star(radius=1.0, mass=mass, ldp=ldp)

    # Set up the planet.
    period = 365 + 30 * np.random.randn()
    size = 0.01 + 0.02 * np.random.rand()
    epoch = period * np.random.rand()
    planet = bart.Planet(size, star.get_semimajor(period), t0=epoch)

    # Set up the system.
    ps = bart.PlanetarySystem(star)
    ps.add_planet(planet)

    # Load the data.
    lcs = kic.get_light_curves(short_cadence=False)
    for lc in lcs:
        print(lc.filename)
        with lc.open() as f:
            # The light curve data are in the first FITS HDU.
            hdu_data = f[1].data
            time = hdu_data["time"]
            flux = hdu_data["sap_flux"]
            ferr = hdu_data["sap_flux_err"]
            quality = hdu_data["sap_quality"]

            inds = ~(np.isnan(time) + np.isnan(flux) + (quality != 0))
            flux[inds] *= ps.lightcurve(time[inds])

            mu = np.median(flux[inds])
            flux /= mu
            ferr /= mu

            # Run untrendy.
            trend = untrendy.fit_trend(time[inds], flux[inds], ferr[inds],
                                       fill_times=10 ** -1.25, dt=5)
            factor = trend(time[inds])

            pl.plot((time[inds] - epoch + 0.5 * period) % period
                    - 0.5 * period,
                    flux[inds] / factor, ".k")

    pl.xlim(-1, 1)
    pl.savefig("injection.png")

if __name__ == "__main__":
    inject(2283589)
