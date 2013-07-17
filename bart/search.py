#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
from . import _george


def search(periods, depth, datasets, alpha=1.0, l2=2.0):
    lnlike = np.empty(len(periods))
    for i, period in enumerate(periods):
        duration = np.exp(0.44 * np.log(period) - 2.97)
        epochs = np.arange(0, period, 0.2 * duration)
        tmp_ll = np.empty(len(epochs))
        for j, t0 in enumerate(epochs):
            ll = 0.0
            N = 0
            for ds in datasets:
                folded = (ds.time - t0 + 0.5 * duration) % period
                data_mask = folded < 4 * duration
                if not np.sum(data_mask):
                    continue
                transit_mask = folded[data_mask] < duration
                model = np.ones(np.sum(data_mask))
                model[transit_mask] *= 1 - depth
                f = ds.flux[data_mask]
                mean = f / model - 1
                ll += (_george.lnlikelihood(ds.time[data_mask], mean,
                                            ds.ferr[data_mask],
                                            alpha, l2)
                       - _george.lnlikelihood(ds.time[data_mask], f - 1,
                                              ds.ferr[data_mask],
                                              alpha, l2))
                N += np.sum(data_mask)
            tmp_ll[j] = ll

        lnlike[i] = np.max(tmp_ll)
        print(period, lnlike[i])

    return lnlike
