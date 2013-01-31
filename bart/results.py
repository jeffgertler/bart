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

from .ldp import QuadraticLimbDarkening


class ResultsProcess(object):

    def __init__(self, filename="mcmc.h5", basepath="."):
        self.basepath = basepath
        self.fn = os.path.join(basepath, filename)
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

    def savefig(self, outfn, fig=None, **kwargs):
        if fig is None:
            fig = pl.gcf()

        from bart import __version__, __commit__
        txt = "Generated using Bart v{0}-{1}".format(__version__, __commit__)
        fig.axes[0].annotate(txt, [1, 1], xycoords="figure fraction",
                            xytext=[-5, -5], textcoords="offset points",
                            ha="right", va="top", fontsize=11)

        # kwargs["dpi"] = kwargs.pop("dpi", 300)

        fn, ext = os.path.splitext(os.path.join(self.basepath, outfn))
        return [fig.savefig(fn + e, **kwargs) for e in
                            [".png"]]

    def itersteps(self):
        for v in self.flatchain:
            self.system.vector = v
            yield self.system

    def _corner_plot(self, outfn, parameters):
        plotchain = np.empty([len(self.flatchain), len(parameters)])
        for i, s in enumerate(self.itersteps()):
            plotchain[i] = np.concatenate([p.getter(self.system)
                                                    for p in parameters])

        # Grab the labels.
        labels = [p.name for p in parameters]

        fig = triangle.corner(plotchain, labels=labels, bins=20)
        self.savefig(outfn, fig=fig)

    def corner_plot(self, parameters, outfn="corner"):
        p = Process(target=self._corner_plot, args=(outfn, parameters))
        p.start()

    def _lc_plot(self, args):
        outdir, planet_ind = args
        time, flux, ivar = self.data

        # Find the period and stellar flux of each sample.
        period = np.empty(len(self.flatchain))
        f = np.empty(len(self.flatchain))

        # Compute the light curve for each sample.
        nt = 5000
        lc = np.empty((len(self.flatchain), nt))
        t = None

        # Loop over the samples.
        for i, s in enumerate(self.itersteps()):
            f[i] = s.star.flux
            period[i] = float(s.planets[planet_ind].get_period(s.star.mass))
            if t is None:
                t = np.linspace(0, 1.5 * period[i], nt)
            lc[i] = s.lightcurve(t) / s.star.flux

        # Compute the stellar flux.
        f0 = np.median(f)
        P0 = np.median(period)

        # Plot the data and samples.
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(time % P0, (flux / f0 - 1) * 1e3, ".k",
                alpha=0.3)
        ax.plot(t, (lc.T - 1) * 1e3, color="#4682b4", alpha=0.1)

        # Annotate the axes.
        ax.set_xlim(0, P0)
        ax.set_xlabel(u"Phase [days]")
        ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")

        self.savefig(os.path.join(outdir, "{0}.png".format(planet_ind)),
                     fig=fig)

    def _lc_plots(self, outdir):
        # Try to make the directory.
        try:
            os.makedirs(outdir)
        except os.error:
            pass

        # Generate the plots.
        map(self._lc_plot,
            [(outdir, i) for i in range(self.system.nplanets)])

    def lc_plot(self, outdir="lightcurves"):
        p = Process(target=self._lc_plots, args=(outdir,))
        p.start()

    def _ldp_plot(self, outfn):
        # Load LDP samples.
        bins, i = self.system.star.ldp.plot()
        ldps = np.empty([len(self.flatchain), len(bins)])
        for i, s in enumerate(self.itersteps()):
            b, intensity = s.star.ldp.plot()
            ldps[i] = intensity

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(bins, ldps.T, "k", alpha=0.1)

        # Over-plot the default Kepler LDP.
        b, ldp = QuadraticLimbDarkening(1000, 0.39, 0.1).plot()
        ax.plot(b, ldp, color="#4682b4", lw=1.5)

        self.savefig(outfn, fig=fig)

    def ldp_plot(self, outfn="ldp"):
        p = Process(target=self._ldp_plot, args=(outfn,))
        p.start()

    def _time_plot(self, outdir):
        fig = pl.figure()
        for i in range(self.chain.shape[2]):
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(self.chain[:, :, i].T)
            self.savefig(os.path.join(outdir, "{0}.png".format(i)), fig=fig)

    def time_plot(self, outdir="time"):
        try:
            os.makedirs(outdir)
        except os.error:
            pass

        p = Process(target=self._time_plot, args=(outdir,))
        p.start()


class Column(object):

    def __init__(self, name, getter=None):
        self.name = name
        if getter is not None:
            self.getter = getter

    def __str__(self):
        return self.name
