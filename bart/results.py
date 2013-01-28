#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["ResultsProcess"]

import os
import cPickle as pickle
from multiprocessing import Process

import h5py
import numpy as np
import matplotlib.pyplot as pl

import triangle


class ResultsProcess(object):

    def __init__(self, fn):
        self.fn = fn
        with h5py.File(self.fn) as f:
            self.system = pickle.loads(str(f["initial_pickle"][...]))

            # Load the data.
            self.data = np.array(f["data"][...])

            # Get and un-pickle the parameter list.
            self.parlist = [pickle.loads(p) for p in f["parlist"][...]]

            self.chain = np.array(f["mcmc"]["chain"][...])
            self.lnprob = np.array(f["mcmc"]["lnprob"][...])

            # Get the ``flatchain`` (see:
            #  https://github.com/dfm/emcee/blob/master/emcee/ensemble.py)
            s = self.chain.shape
            self.flatchain = self.chain.reshape(s[0] * s[1], s[2])

    def _corner_plot(self, outfn):
        # Construct the list of samples to plot.
        plotchain = []  # np.empty(self.flatchain.shape)
        i = 0
        for p in self.parlist:
            if p.plot_results:
                plotchain.append(p.iconv(self.flatchain[:, i:i + len(p)]))
                i += len(p)
        plotchain = np.concatenate(plotchain, axis=-1)
        plotchain = np.hstack([plotchain,
                               np.atleast_2d(self.lnprob.flatten()).T])

        # Grab the labels.
        labels = np.concatenate([p.names for p in self.parlist]
                                + [["ln-prob"]])

        fig = triangle.corner(plotchain, labels=labels, bins=20)
        fig.savefig(outfn)

    def corner_plot(self, outfn="./corner.png"):
        p = Process(target=self._corner_plot, args=(outfn,))
        p.start()

    def _lc_plot(self, args):
        outdir, planet_ind = args
        time, flux, ivar = self.data

        # Access the planet and star in the planetary system.
        planet = self.system.planets[planet_ind]
        star = self.system.star

        # Find the period and stellar flux of each sample.
        period = np.empty(len(self.flatchain))
        f = np.empty(len(self.flatchain))

        # Compute the light curve for each sample.
        t = np.linspace(time.min(), time.max(), 5000)
        lc = np.empty((len(self.flatchain), len(t)))

        # Loop over the samples.
        for i, v in enumerate(self.flatchain):
            self.system.vector = v
            f[i] = star.flux
            period[i] = float(planet.get_period(star.mass))
            lc[i] = self.system.lightcurve(t) / star.flux

        # Fold the samples.
        T = np.median(period)
        t = t % T
        inds = np.argsort(t)
        lc = lc[:, inds]
        t = t[inds]

        # Compute the stellar flux.
        f0 = np.median(f)

        # Plot the data and samples.
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(time % T, (flux / f0 - 1) * 1e3, ".k", alpha=0.3)
        ax.plot(t, (lc.T - 1) * 1e3, color="k", alpha=0.1)

        # Annotate the axes.
        ax.set_xlim(0, T)
        ax.set_xlabel(u"Phase [days]")
        ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")

        fig.savefig(os.path.join(outdir, "{0}.png".format(planet_ind)))

    def _lc_plots(self, outdir):
        # Try to make the directory.
        try:
            os.makedirs(outdir)
        except os.error:
            pass

        # Generate the plots.
        map(self._lc_plot,
            [(outdir, i) for i in range(self.system.nplanets)])

    def lc_plot(self, outdir="./lightcurves"):
        p = Process(target=self._lc_plots, args=(outdir,))
        p.start()

    def _time_plot(self, outdir):
        fig = pl.figure()
        for i in range(self.chain.shape[2]):
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(self.chain[:, :, i].T)
            fig.savefig(os.path.join(outdir, "{0}.png".format(i)))

    def time_plot(self, outdir="./time"):
        try:
            os.makedirs(outdir)
        except os.error:
            pass

        p = Process(target=self._time_plot, args=(outdir,))
        p.start()
