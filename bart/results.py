#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import cPickle as pickle
from multiprocessing import Process

import h5py
import numpy as np
import matplotlib.pyplot as pl

import triangle

# from .bart import PlanetarySystem  # NOQA


class ResultsProcess(object):

    def __init__(self, fn):
        self.fn = fn
        with h5py.File(self.fn) as f:
            self.system = pickle.loads(str(f["initial_pickle"][...]))

            # Load the data.
            self.data = np.array(f["data"][...])

            # Get and un-pickle the parameter list.
            self.parlist = f["parlist"][...]
            self.parlist = [pickle.loads(p[1]) for p in self.parlist]

            self.chain = np.array(f["mcmc"]["chain"][...])

            # Get the ``flatchain`` (see:
            #  https://github.com/dfm/emcee/blob/master/emcee/ensemble.py)
            s = self.chain.shape
            self.chain = self.chain.reshape(s[0] * s[1], s[2])

    def _corner_plot(self, outfn):
        # Construct the list of samples to plot.
        plotchain = np.empty(self.chain.shape)
        for i, p in enumerate(self.parlist):
            plotchain[:, i] = p.iconv(self.chain[:, i])
            plotchain[:, i] -= np.median(plotchain[:, i])

        # Grab the labels.
        labels = [p.name for p in self.parlist]

        fig = triangle.corner(plotchain, labels=labels, bins=20)
        fig.savefig(outfn)

    def corner_plot(self, outfn="./corner.png"):
        p = Process(target=self._corner_plot, args=(outfn,))
        p.start()

    def _lc_plot(self, outfn):
        time, flux, ivar = self.data
