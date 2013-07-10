#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["kepler_injection"]

import numpy as np

try:
    import kplr
    kplr = kplr
except ImportError:
    kplr = None
else:
    from kplr.ld import get_quad_coeffs

from . import data, ld, Star, Planet, PlanetarySystem


def default_cleaner(time, flux, ferr, quality):
    mask = (np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)
            * (quality == 0))
    return [v[mask] for v in [time, flux, ferr]]


def kepler_injection(kicid, period, size, b=0.0, pdc=False,
                     cleaner=default_cleaner, **planet_params):
    """
    Inject a transit into a particular KIC target. This function depends on
    the `kplr <http://dan.iel.fm/kplr>`_ module to download the raw
    light curves from MAST.

    :param kicid:
        The index of the target in the KIC.

    :param period:
        The period of the orbit in days.

    :param size:
        The radius of the planet relative to the star.

    :param b:
        The impact parameter of the observed transit in stellar radii.
        (default: 0.0)

    :param pdc:
        Should the transit be injected into the PDC data? If not, use SAP.
        (default: False)

    :param cleaner:
        A function that performs any data munging that you want. The default
        just masks NaNs and removes any data with non-zero quality mask. The
        function should take 4 (numpy array) arguments: time, flux,
        uncertainty, and quality. It should return 3 arrays: time, flux, and
        uncertainty.

    :param **planet_params:
        Any other keyword arguments to pass when building the :class:`Planet`.

    :returns datasets:
        A list of :class:`data.GPLightCurve` datasets with the transit
        injected into the flux.

    :returns ps:
        The :class:`PlanetarySystem` used to generate the transit.

    """
    if kplr is None:
        raise ImportError("You need to install kplr (http://dan.iel.fm/kplr) "
                          "before doing a Kepler injection.")

    # Get the stellar information from the API.
    client = kplr.API()
    kic = client.star(kicid)
    teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
    assert teff is not None

    # Get an approximate limb darkening law.
    mu1, mu2 = get_quad_coeffs(teff, logg=logg, feh=feh)
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

    # Build the star object.
    star = Star(ldp=ldp)

    # Figure out the semi-major axis associated to the given period.
    a = star.get_semimajor(period)

    # Figure out the inclination based on the impact parameter.
    planet_params["ix"] = 90.0 - np.degrees(np.arctan2(a, b))

    # Set up the planet.
    planet = Planet(size, a, **planet_params)

    # Update the period if it wasn't given as input.
    if period is None:
        period = planet.get_period(star.mass)

    # Set up the system.
    ps = PlanetarySystem(star)
    ps.add_planet(planet)

    # Download the data and inject transits.
    if pdc:
        cols = ["time", "pdcsap_flux", "pdcsap_flux_err", "sap_quality"]
    else:
        cols = ["time", "sap_flux", "sap_flux_err", "sap_quality"]
    datasets = []
    lcs = kic.get_light_curves(short_cadence=False, fetch=False)
    for lc in lcs:
        with lc.open() as f:
            # The light curve data are in the first FITS HDU.
            hdu_data = f[1].data
            time, flux, ferr, quality = [hdu_data[k] for k in cols]

        # Inject the transit.
        m = np.isfinite(time) * np.isfinite(flux)
        flux[m] *= ps.lightcurve(time[m])

        # Clean the data using the provided callback and save the dataset.
        time, flux, ferr = cleaner(time, flux, ferr, quality)
        datasets.append(data.GPLightCurve(time, flux, ferr))

    return datasets, ps
