#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["inject"]

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

output_path = "data"


def inject(kicid):
    np.random.seed(int(kicid))

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
    star = bart.Star(radius=radius, mass=mass, ldp=ldp)

    # Set up the planet.
    period = 365 + 30 * np.random.randn()
    size = 0.01 + 0.03 * np.random.rand()
    epoch = period * np.random.rand()
    planet = bart.Planet(size, star.get_semimajor(period), t0=epoch)
    print(radius, period, size, epoch)

    # Set up the system.
    ps = bart.PlanetarySystem(star)
    ps.add_planet(planet)

    # Make sure that that data directory exists.
    base_dir = os.path.join(output_path, "{0}".format(kicid))
    try:
        os.makedirs(base_dir)
    except os.error:
        pass

    # Load the data and inject into each transit.
    lcs = kic.get_light_curves(short_cadence=False)
    for lc in lcs:
        print(lc.filename)
        with lc.open() as f:
            # The light curve data are in the first FITS HDU.
            hdu_data = f[1].data
            time = hdu_data["time"]
            sap_flux = hdu_data["sap_flux"]
            sap_ferr = hdu_data["sap_flux_err"]
            quality = hdu_data["sap_quality"]

            inds = ~(np.isnan(time) + np.isnan(sap_flux))
            sap_flux[inds] *= ps.lightcurve(time[inds])

            inds *= quality == 0
            flux = np.array(sap_flux)
            ferr = np.array(sap_ferr)

            mu = np.median(flux[inds])
            flux /= mu
            ferr /= mu

            # Run untrendy.
            trend = untrendy.fit_trend(time[inds], flux[inds], ferr[inds],
                                       fill_times=10 ** -1.25, dt=10)
            factor = trend(time[inds])
            flux[inds] /= factor

            pl.plot((time[inds] - epoch + 0.5 * period) % period
                    - 0.5 * period,
                    flux[inds], ".k")

        # Coerce the filename.
        fn = os.path.splitext(os.path.split(lc.filename)[1])[0] + ".txt"
        with open(os.path.join(base_dir, fn), "w") as f:
            for line in zip(time, sap_flux, sap_ferr, flux, ferr, quality):
                f.write(", ".join(map("{0}".format, line)) + "\n")

    pl.xlim(-5, 5)
    pl.savefig("injection.png")

if __name__ == "__main__":
    inject(2283589)
