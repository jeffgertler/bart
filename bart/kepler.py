#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["load", "fiducial_ldp"]

import pyfits
import numpy as np

from .ldp import QuadraticLimbDarkening


def load(fn):
    """
    Load, normalize and sanitize a Kepler light curve downloaded from MAST.
    The median time will be subtracted and the median flux is divided out of
    the flux and flux uncertainty for numerical stability.

    :param fn:
        The path to the FITS file.

    :return time:
        The times of the samples.

    :return flux:
        The brightness at the time samples.

    :return ferr:
        The quoted uncertainty on the flux.

    """
    f = pyfits.open(fn)
    lc = np.array(f[1].data)
    f.close()

    time = lc["TIME"]
    flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    # t0 = int(np.median(time[~np.isnan(time)]))
    # time = time - t0

    mu = np.median(flux[~np.isnan(flux)])
    flux /= mu
    ferr /= mu

    return (time, flux, ferr)


def fiducial_ldp(bins=100):
    try:
        nbins = len(bins)
    except TypeError:
        nbins = int(bins)
        bins = None
    ldp = QuadraticLimbDarkening(nbins, 0.39, 0.1)
    if bins is not None:
        ldp.bins = bins
    return ldp
