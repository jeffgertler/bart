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

        # print(self.parlist)

        # Get the ``flatchain`` (see:
        #  https://github.com/dfm/emcee/blob/master/emcee/ensemble.py)
        s = self.chain.shape
        self.flatchain = self.chain.reshape(s[0] * s[1], s[2])

        # Pre-process the chain to get the median system.
        spec = np.empty([len(self.flatchain), len(self.system.spec)])
        for i, s in enumerate(self.itersteps()):
            spec[i] = s.spec
        self.median_spec = np.median(spec, axis=0)
        self.system.spec = self.median_spec
        print(self.median_spec)

        # Find the median stellar parameters.
        self.fstar = self.system.star.flux
        self.rstar = self.system.star.radius
        self.mstar = self.system.star.mass

        # Find the median planet parameters.
        self.periods = [p.get_period(self.system.star.mass)
                        for p in self.system.planets]
        self.radii = [p.r for p in self.system.planets]
        self.epochs = [p.t0 for p in self.system.planets]
        self.semimajors = [p.a for p in self.system.planets]

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
                            [".png", ".pdf"]]

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

        # Get the median parameters of the fit.
        fstar, rstar, mstar = self.fstar, self.rstar, self.mstar
        P, t0, a, r = (self.periods[planet_ind], self.epochs[planet_ind],
                       self.semimajors[planet_ind], self.radii[planet_ind])

        # Compute the transit duration.
        duration = P * (r + rstar) / np.pi / a
        t = np.linspace(-duration, duration, 5000)

        # Compute the light curve for each sample.
        lc = np.empty((len(self.flatchain), len(t)))

        # Loop over the samples.
        for i, s in enumerate(self.itersteps()):
            s.planets[planet_ind].t0 = 0.0
            lc[i] = s.lightcurve(t)

        # Plot the data and samples.
        fig = pl.figure()
        ax = fig.add_subplot(111)
        time = time % P - t0
        inds = (time < duration) * (time > -duration)
        ax.plot(time[inds], (flux[inds] / fstar - 1) * 1e3, ".",
                alpha=1.0, color="#888888")
        ax.plot(t, (lc.T / fstar - 1) * 1e3, color="#000000", alpha=0.03)

        # Annotate the axes.
        ax.set_xlim(-duration, duration)
        ax.set_xlabel(u"Phase [days]")
        ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")

        self.savefig(os.path.join(outdir, "{0}.png".format(planet_ind)),
                     fig=fig)

    def _lc_plots(self, outdir):
        # Try to make the directory.
        try:
            os.makedirs(os.path.join(self.basepath, outdir))
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
        ldps = np.empty([len(self.flatchain), i.shape[0], i.shape[1]])
        for i, s in enumerate(self.itersteps()):
            b, intensity = s.star.ldp.plot()
            ldps[i] = intensity

        fig = pl.figure()
        ax = fig.add_subplot(111)
        [ax.plot(bins[i], ldps[:, i].T, "k", alpha=0.1)
                                        for i in range(len(bins))]

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
            # ax.set_title(self.parlist[i].name)
            self.savefig(os.path.join(outdir, "{0}.png".format(i)), fig=fig)

    def time_plot(self, outdir="time"):
        try:
            os.makedirs(os.path.join(self.basepath, outdir))
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
