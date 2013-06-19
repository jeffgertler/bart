#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["inject"]

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart
import kplr
from kplr.ld import LDCoeffAdaptor

adapter = LDCoeffAdaptor(model="claret11")

client = kplr.API()

output_path = "data"


def inject(kicid):
    print(kicid)
    np.random.seed(int(kicid))

    # Get the KIC entry.
    kic = client.star(kicid)
    teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
    assert teff is not None

    # Get the limb darkening law.
    mu1, mu2 = adapter.get_coeffs(teff, logg=logg, feh=feh)
    if np.any(np.isnan([mu1, mu2])):
        mu1, mu2 = adapter.get_coeffs(teff)
        if np.any(np.isnan([mu1, mu2])):
            return None
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

    # Build the star object.
    star = bart.Star(ldp=ldp)

    # Set up the planet.
    period = 365 + 30 * np.random.randn()
    size = 0.005 + 0.02 * np.random.rand()
    epoch = period * np.random.rand()
    incl = np.degrees(np.acos(0.01 * np.random.randn()))
    a = star.get_semimajor(period)
    b = a / np.tan(np.radians(incl))
    planet = bart.Planet(size, a, t0=epoch)

    # Set up the system.
    ps = bart.PlanetarySystem(star, iobs=incl)
    ps.add_planet(planet)

    # Make sure that that data directory exists.
    base_dir = os.path.join(output_path, "{0}".format(kicid))
    try:
        os.makedirs(base_dir)
    except os.error:
        pass

    # Save the injected values.
    with open(os.path.join(base_dir, "truth.txt"), "w") as f:
        f.write("# " + ",".join(["r/R", "b", "incl", "period", "epoch"])
                + "\n")
        f.write(",".join(map("{0}".format, [size, b, incl, period, epoch])))
        f.write("\n")

    # Load the data and inject into each transit.
    lcs = kic.get_light_curves(short_cadence=False)
    for lc in lcs:
        print(lc.filename)
        with lc.open() as f:
            # The light curve data are in the first FITS HDU.
            hdu_data = f[1].data

            # Load the data columns.
            time = hdu_data["time"]
            sap_flux = hdu_data["sap_flux"]
            sap_ferr = hdu_data["sap_flux_err"]
            pdc_flux = hdu_data["pdcsap_flux"]
            pdc_ferr = hdu_data["pdcsap_flux_err"]
            quality = hdu_data["sap_quality"]

            # Mask out missing data.
            inds = ~(np.isnan(time) + np.isnan(sap_flux))
            pdcinds = ~(np.isnan(time) + np.isnan(pdc_flux))

            # Inject the transit into the SAP and PDC light curves.
            sap_flux[inds] *= ps.lightcurve(time[inds])
            pdc_flux[pdcinds] *= ps.lightcurve(time[pdcinds])

        # Coerce the filename.
        fn = os.path.splitext(os.path.split(lc.filename)[1])[0] + ".txt"
        with open(os.path.join(base_dir, fn), "w") as f:
            f.write("# " + ",".join(["time", "sap_flux", "sap_ferr",
                                     "pdc_flux", "pdc_ferr", "quality"])
                    + "\n")
            for line in zip(time, sap_flux, sap_ferr, pdc_flux, pdc_ferr,
                            quality):
                f.write(",".join(map("{0}".format, line)) + "\n")

    return None


if __name__ == "__main__":
    from multiprocessing import Pool

    # Load a list of KIC targets.
    with open("q11_solar_type.dat") as f:
        targets = [int(k) for k in f.readlines()]

    ss = 100
    for i in range(3540, 5000, ss):
        pool = Pool()
        try:
            pool.map(inject, targets[i:i + ss])
        except:
            print("Failed.")
