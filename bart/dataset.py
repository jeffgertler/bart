#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Dataset", "KeplerDataset", "RVDataset"]

import pyfits
import numpy as np
from .bart import Model
from . import kepler


class Dataset(Model):

    __type__ = "lc"

    def __init__(self, time, flux, ferr, texp, zp=1.0, jitter=0.0):
        super(Dataset, self).__init__()

        self.texp = texp
        self.jitter = 0.0
        self.zp = zp

        # Sanitize the data.
        inds = ~np.isnan(time) * ~np.isnan(flux) * ~np.isnan(ferr)
        self.time, self.flux, self.ferr = time[inds], flux[inds], ferr[inds]
        self.ivar = 1.0 / self.ferr / self.ferr


class KeplerDataset(Dataset):

    def __init__(self, fn, jitter=0.0):
        f = pyfits.open(fn)
        lc = np.array(f[1].data)
        self.cadence = 0 if f[0].header["OBSMODE"] == "short cadence" else 1
        f.close()

        # Get the exposure time.
        # http://archive.stsci.edu/mast_faq.php?mission=KEPLER#50
        texp = kepler.EXPOSURE_TIMES[self.cadence]

        time = lc["TIME"]  # + t0
        flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

        super(KeplerDataset, self).__init__(time, flux, ferr, texp / 60.,
                                            jitter=jitter)

        # Remove the arbitrary median.
        self.median = np.median(self.flux)
        self.flux /= self.median
        self.ferr /= self.median
        self.ivar *= self.median * self.median


class RVDataset(Model):

    __type__ = "rv"

    def __init__(self, time, rv, rverr, jitter=0.0):
        super(RVDataset, self).__init__()
        inds = ~np.isnan(time) * ~np.isnan(rv) * ~np.isnan(rverr)
        self.time = time[inds] - kepler.TIME_ZERO
        self.rv = rv[inds]
        self.rverr = rverr[inds]
        self.ivar = 1.0 / self.rverr / self.rverr
        self.jitter = jitter
