#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Dataset", "KeplerDataset", "RVDataset"]

import pyfits
import numpy as np


class Dataset(object):

    __type__ = "lc"

    def __init__(self, time, flux, ferr, texp):
        self.texp = texp

        # Sanitize the data.
        inds = ~np.isnan(time) * ~np.isnan(flux) * ~np.isnan(ferr)
        self.time, self.flux, self.ferr = time[inds], flux[inds], ferr[inds]
        self.ivar = 1.0 / self.ferr / self.ferr


class KeplerDataset(Dataset):

    def __init__(self, fn):
        f = pyfits.open(fn)
        lc = np.array(f[1].data)
        cadence = 0 if f[0].header["OBSMODE"] == "short cadence" else 1
        f.close()

        # Get the exposure time.
        # http://archive.stsci.edu/mast_faq.php?mission=KEPLER#50
        texp = [54.2, 1626][cadence]

        time = lc["TIME"]
        flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

        super(KeplerDataset, self).__init__(time, flux, ferr, texp / 60.)

        # Remove the arbitrary median.
        self.median = np.median(self.flux)
        self.flux /= self.median
        self.ferr /= self.median
        self.ivar *= self.median * self.median


class RVDataset(Dataset):

    __type__ = "rv"

    def __init__(self, time, rv, rverr):
        inds = ~np.isnan(time) * ~np.isnan(rv) * ~np.isnan(rverr)

        self.time = time[inds]
        self.rv = rv[inds]
        self.rverr = rverr[inds]
        self.ivar = 1.0 / self.rverr / self.rverr
