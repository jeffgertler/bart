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
import pystache

import triangle


class ResultsProcess(object):

    def __init__(self, filename="mcmc.h5", basepath=".", thin=1, burnin=0,
                 subsamples=12):
        self.subsamples = subsamples
        self.burnin = burnin
        self.thin = thin
        self.basepath = basepath
        self.fn = os.path.join(basepath, filename)
        with h5py.File(self.fn) as f:
            self.system = pickle.loads(str(f["initial_pickle"][...]))

            # Load the data.
            self.datasets = pickle.loads(str(f["datasets"][...]))

            # Get and un-pickle the parameter list.
            self.parlist = [pickle.loads(p) for p in f["parlist"][...]]

            self._chain = np.array(f["mcmc"]["chain"][...])
            self._lnprob = np.array(f["mcmc"]["lnprob"][...])

        self.chain = self._chain[:, burnin::thin, :]
        self.lnprob = self._lnprob[:, burnin::thin]

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

        fn, ext = os.path.splitext(os.path.join(self.basepath, outfn))
        return [fig.savefig(fn + e, **kwargs) for e in
                            [".png", ".pdf"]]

    def itersteps(self, thin=None):
        if thin is None:
            thin = self.thin
        for v in self.flatchain:
            self.system.vector = v
            yield self.system

    def _corner_plot(self, outfn, parameters):
        plotchain = np.empty([len(self.flatchain), len(parameters)])
        for i, s in enumerate(self.itersteps()):
            plotchain[i] = np.array([float(p.getter(self.system))
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
        time = np.concatenate([d.time for d in self.datasets
                                      if d.__type__ == "lc"])
        flux = np.concatenate([d.flux for d in self.datasets
                                      if d.__type__ == "lc"])
        texp = np.min([d.texp for d in self.datasets
                              if d.__type__ == "lc"])

        # Get the median parameters of the fit.
        fstar, rstar, mstar = self.fstar, self.rstar, self.mstar
        P, t0, a, r = (self.periods[planet_ind], self.epochs[planet_ind],
                       self.semimajors[planet_ind], self.radii[planet_ind])

        # Compute the transit duration.
        duration = P * (r + rstar) / np.pi / a
        t = np.linspace(-duration, duration, 5000)

        # Compute the light curve for each sample.
        lc = np.empty((self.subsamples, len(t)))

        # Loop over the samples.
        inds = np.random.randint(len(self.flatchain), size=self.subsamples)
        for i, v in enumerate(self.flatchain[inds, :]):
            self.system.vector = v
            self.system.planets[planet_ind].t0 = 0.0
            lc[i] = self.system.lightcurve(t, texp=texp)

        # Plot the data and samples.
        fig = pl.figure()
        ax = fig.add_subplot(111)
        time = time % P - t0
        inds = (time < duration) * (time > -duration)
        ax.plot(time[inds] * 24.0, flux[inds] / fstar, ".",
                alpha=1.0, color="#888888", rasterized=True)
        ax.plot(t * 24.0, lc.T / fstar, color="k", alpha=0.5)

        # Annotate the axes.
        ax.set_xlim(-duration * 24.0, duration * 24.0)
        ax.set_xlabel(u"Time [Hours Since Transit]")
        ax.set_ylabel(r"Flux")

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

    def _rv_plot(self, args):
        outdir, planet_ind = args
        try:
            time = np.concatenate([d.time for d in self.datasets
                                          if d.__type__ == "rv"])
            rv_data = np.concatenate([d.rv for d in self.datasets
                                          if d.__type__ == "rv"])
            rverr_data = np.concatenate([d.rverr for d in self.datasets
                                                if d.__type__ == "rv"])
        except ValueError:
            time, rv_data, rverr_data = 3 * [np.array([])]

        # Get the median parameters of the fit.
        fstar, rstar, mstar = self.fstar, self.rstar, self.mstar
        P, t0, a, r = (self.periods[planet_ind], self.epochs[planet_ind],
                       self.semimajors[planet_ind], self.radii[planet_ind])

        # Compute the light curve for each sample.
        t = np.linspace(0, P, 5000)
        rv = np.empty((self.subsamples, len(t)))

        # Loop over the samples.
        inds = np.random.randint(len(self.flatchain), size=self.subsamples)
        for i, v in enumerate(self.flatchain[inds, :]):
            self.system.vector = v
            rv[i] = self.system.radial_velocity(t)

        # Plot the data and samples.
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.errorbar((time % P) * 24.0, rv_data, yerr=rverr_data, fmt=".",
                alpha=1.0, color="#888888")
        ax.plot(t * 24.0, rv.T, color="k", alpha=0.5)

        # Annotate the axes.
        ax.set_xlabel(u"Time")
        ax.set_ylabel(r"Velocity [m s$^{-1}$]")

        self.savefig(os.path.join(outdir, "{0}.png".format(planet_ind)),
                     fig=fig)

        return ax

    def _rv_plots(self, outdir):
        # Try to make the directory.
        try:
            os.makedirs(os.path.join(self.basepath, outdir))
        except os.error:
            pass

        # Generate the plots.
        return map(self._rv_plot,
            [(outdir, i) for i in range(self.system.nplanets)])

    def rv_plot(self, outdir="rv"):
        p = Process(target=self._rv_plots, args=(outdir,))
        p.start()

    def _ldp_plot(self, outfn, fiducial):
        N = self.subsamples

        # Load LDP samples.
        bins, i = self.system.star.ldp.plot()
        ldps = np.empty([N, i.shape[0], i.shape[1]])
        inds = np.random.randint(len(self.flatchain), size=N)
        for i, v in enumerate(self.flatchain[inds, :]):
            self.system.vector = v
            b, intensity = self.system.star.ldp.plot()
            ldps[i] = intensity

        fig = pl.figure()
        ax = fig.add_subplot(111)
        [ax.plot(bins.flatten() + 0.005 * np.random.randn(),
                 l.flatten(), color="k", alpha=0.5) for l in ldps]
        # [ax.plot(bins[i], ldps[:, i].T, "k", alpha=0.1)
        #                                 for i in range(len(bins))]

        # Over-plot the default Kepler LDP.
        if fiducial is not None:
            rs = np.linspace(0, 1, 1000)
            ldp = fiducial(rs) / fiducial.norm
            ax.plot(rs, ldp, "--", color="#4682b4", lw=3)

        ax.set_xlim(0, 1)
        d = ldps.max() - ldps.min()
        ax.set_ylim(ldps.min() - 0.1 * d, ldps.max() + 0.1 * d)

        ax.set_xlabel("$r / R_\star$")
        ax.set_ylabel("$I(r)$")

        self.savefig(outfn, fig=fig)

    def ldp_plot(self, outfn="ldp", fiducial=None):
        p = Process(target=self._ldp_plot, args=(outfn, fiducial))
        p.start()

    def _time_plot(self, outdir):
        fig = pl.figure()
        names = np.concatenate([p.names for p in self.parlist])
        for i in range(self._chain.shape[2]):
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(self._chain[:, :, i].T)
            ax.axvline(self.burnin, color="k", ls="dashed")
            ax.set_title(names[i])
            fig.savefig(os.path.join(self.basepath, outdir,
                                     "{0}.png".format(names[i].strip("$"))))

        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self._lnprob.T)
        ax.axvline(self.burnin, color="k", ls="dashed")
        ax.set_title("ln-prob")
        fig.savefig(os.path.join(self.basepath, outdir,
                                    "lnprob.png".format(i)))

    def time_plot(self, outdir="time"):
        try:
            os.makedirs(os.path.join(self.basepath, outdir))
        except os.error:
            pass

        p = Process(target=self._time_plot, args=(outdir,))
        p.start()

    def latex(self, parameters, outfn="table.tex", title="",
              caption="", refs=""):
        # Load the template.
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "table_temp.tex")
        tmp = open(fn).read()

        plotchain = np.empty([len(self.flatchain), len(parameters)])
        for i, s in enumerate(self.itersteps()):
            plotchain[i] = np.concatenate([p.getter(self.system)
                                                    for p in parameters])

        # Grab the labels.
        labels = [p.name for p in parameters]
        body = ""
        for i, vals in enumerate(plotchain.T):
            vals = np.sort(vals)
            mn, md, mx = (vals[int(0.16 * len(vals))],
                          vals[int(0.5 * len(vals))],
                          vals[int(0.84 * len(vals))])

            # Compute the precision needed in the quantiles.
            p = min(int(np.log10(md - mn)), int(np.log10(mx - md)))
            if p < 0:
                p = -p + 2
            else:
                p = 2
            fmt = "{{0:.{0}f}}".format(p)

            body += labels[i] + " & "
            body += " & ".join([fmt.format(m) for m in [mn, md, mx]])
            body += " \\\\\n"

        with open(os.path.join(self.basepath, outfn), "w") as f:
            f.write(pystache.render(tmp, {"body": body, "title": title,
                                          "caption": caption, "refs": refs}))


class Column(object):

    def __init__(self, name, getter=None):
        self.name = name
        if getter is not None:
            self.getter = getter

    def __str__(self):
        return self.name
