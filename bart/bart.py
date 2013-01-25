#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["Star", "Planet", "PlanetarySystem"]

from multiprocessing import Process
import cPickle as pickle

import numpy as np
import emcee
import triangle

try:
    import matplotlib.pyplot as pl
    pl = pl
except ImportError:
    pl = None

try:
    import h5py
    h5py = h5py
except ImportError:
    h5py = None

from bart import _bart, mog
from bart.ldp import LimbDarkening


_G = 2945.4625385377644


class Model(object):

    def __init__(self):
        self.parameters = []

    @property
    def vector(self):
        try:
            return np.concatenate([np.atleast_1d(p.getter(self))
                                                for p in self.parameters
                                                if len(p) > 0])
        except ValueError:
            return np.array([])

    @vector.setter  # NOQA
    def vector(self, v):
        self._set_vector(v)

    def _set_vector(self, v):
        i = 0
        for p in self.parameters:
            p.setter(self, v[i:i + len(p)])
            i += len(p)

    def __len__(self):
        return np.sum([len(p) for p in self.parameters])


class Star(Model):
    """
    Represents the parameters of a star that hosts a planetary system.

    :param mass: (optional)
        The mass of the star in Solar masses.

    :param radius: (optional)
        The radius of the star in Solar radii.

    :param flux: (optional)
        The flux of the star in arbitrary units.

    :param ldp: (optional)
        The limb-darkening profile (a subclass of :class:`LimbDarkening`) of
        the star. This will default to a uniformly illuminated star.

    """

    def __init__(self, mass=1.0, radius=1.0, flux=1.0, ldp=None):
        super(Star, self).__init__()

        self.mass = mass
        self.radius = radius
        self.flux = flux

        # The limb darkening profile.
        if ldp is None:
            # Default to a uniformly illuminated star.
            self.ldp = LimbDarkening(1.0, 1.0)
        else:
            self.ldp = ldp


class Planet(Model):
    """
    A :class:`Planet` object represents the set of parameters describing a
    single planet in a planetary system.

    :param r:
        The size of the planet in Solar radii.

    :param a:
        The semi-major axis of the orbit in Solar radii.

    :param t0: (optional)
        The time of a reference pericenter passage.

    :param e: (optional)
        The eccentricity of the orbit.

    :param pomega: (optional)
        The rotation of the orbital ellipse in the reference plane.

    :param ix: (optional)
        The inclination of the orbit around the perpendicular axis to the
        observer's line-of-sight in degrees.

    :param iy: (optional)
        The inclination of the orbit around the observes line-of-sight in
        degrees.

    """

    def __init__(self, r, a, t0=0.0, e=0.0, pomega=0.0, ix=0.0, iy=0.0):
        super(Planet, self).__init__()

        self.r = r
        self.a = a
        self.t0 = t0
        self.e = e
        self.pomega = pomega
        self.ix = ix
        self.iy = iy

    def get_mstar(self, T):
        """
        Get the mass of the host star implied by the semi-major axis of this
        planet and an input period.

        """
        a = self.a
        return a * a * a * 4 * np.pi * np.pi / _G / T / T

    def get_period(self, mstar):
        """
        Get the period of this planet orbiting a star with a given mass.

        """
        a = self.a
        return 2 * np.pi * np.sqrt(a * a * a / _G / mstar)


class PlanetarySystem(Model):
    """
    A system of planets orbiting a star.

    :param star:
        A :class:`Star` object.

    :param iobs: (optional)
        The viewing angle in degrees. 90 is defined as edge on.

    """

    def __init__(self, star, iobs=90.0):
        super(PlanetarySystem, self).__init__()

        self._data = None

        # The properties of the system as a whole.
        self.star = star
        self.iobs = iobs

        # The planets.
        self.planets = []

        # Process management book keeping.
        self._processes = []

    def __del__(self):
        if hasattr(self, u"_processes"):
            for p in self._processes:
                p.join()

    @property
    def nplanets(self):
        """
        The number of planets in the system.

        """
        return len(self.planets)

    def add_planet(self, planet):
        """
        Add a :class:`Planet` to the system.

        :param planet:
            The :class:`Planet` to add.

        """
        self.planets.append(planet)

    @property
    def vector(self):
        """
        Get the list of parameter values for the system, the star, and the
        planets.

        """
        v = super(PlanetarySystem, self).vector
        return np.concatenate([v, self.star.vector]
                              + [p.vector for p in self.planets])

    @vector.setter  # NOQA
    def vector(self, v):
        """
        Set the parameters of the system, the star, and the planets.

        :param v:
            A ``numpy`` array of the target parameter values.

        """
        j, i = len(self), len(self) + len(self.star)
        super(PlanetarySystem, self)._set_vector(v[:j])
        self.star.vector = v[j:i]
        for p in self.planets:
            p.vector = v[i:i + len(p)]
            i += len(p)

    def __call__(self, p):
        return self.lnprob(p)

    def lnprob(self, p):
        """
        Compute the log-probability of the model evaluated at a given position.

        :param p:
            The vector of fit parameters where the model should be evaluated.

        """
        # Make sure that we catch all the over/under-flows.
        np.seterr(all=u"raise")
        try:
            self.vector = p

            # Compute the prior.
            lnp = self.lnprior()
            if np.isinf(lnp) or np.isnan(lnp):
                return -np.inf

            # Compute the likelihood.
            lnl = self.lnlike()
            if np.isinf(lnl) or np.isnan(lnl):
                return -np.inf

        except FloatingPointError:
            return -np.inf

        return lnl + lnp

    def lnprior(self):
        """
        Compute the log-prior of the current model.

        """
        lnp = 0.0
        return lnp

    def lnlike(self):
        """
        Compute the log-likelihood of the current model.

        """
        return -0.5 * self.chi2()

    def chi2(self):
        assert self._data is not None
        model = self.lightcurve(self._data[0])
        delta = self._data[1] - model

        # Add in the jitter.
        ivar = np.array(self._data[2])
        # inds = ivar > 0
        # ivar[inds] = 1. / (1. / ivar[inds] + self.jitter)

        return np.sum(delta * delta * ivar) - np.sum(np.log(ivar))

    def lightcurve(self, t):
        """
        Get the light curve of the model at the current model.

        :param t:
            The times where the light curve should be evaluated.

        """
        s = self.star
        r = [(p.r, p.a, p.t0, p.e, p.pomega, p.ix, p.iy) for p in self.planets]
        r, a, t0, e, pomega, ix, iy = zip(*r)
        ldp = self.star.ldp
        return _bart.lightcurve(t, s.flux, s.mass, s.radius, self.iobs,
                                r, a, t0, e, pomega, ix, iy,
                                ldp.bins, ldp.intensity)

    def _prepare_data(self, t, f, ferr):
        """
        Sanitize some light curve data.

        """
        # Deal with masked and problematic data points.
        inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
            + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
            + (f < 0) + (ferr <= 0))
        t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

        # Store the data.
        self._data = [t, f, ivar]

    def fit(self, data, iterations, start=None, filename="./mcmc.h5",
            **kwargs):
        """
        Fit the data using MCMC to get constraints on the parameters.

        :param data:
            A tuple of the form ``(t, f, ferr)`` giving the data to fit.

        :param iterations:
            The number of MCMC steps to run in the production pass.

        :param filename: (optional)
            The name of the file where the result should be stored.

        :param thin: (optional)
            The number of steps between each saved sample.

        :param start: (optional)
            To start the chain from a specific position, set ``start`` to the
            position. It should have the shape ``(nwalkers, ndim)``.

        :param threads: (optional)
            The number of threads to use for multiprocessing with ``emcee``.

        :param burnin: (optional)
            The burn-in schedule to use. Set this to ``[]`` for no burn-in.
            Otherwise, it should have the form ``[nburn1, nburn2, ...]``. For
            this example, we would run a burn-in chain with ``nburn1`` steps
            and then re-sample the "bad" walkers. Then, repeat with ``nburn2``
            steps.

        :param K: (optional)
            The number of clusters to use for K-means in the trimming step.

        """
        # Check that the vector conversions work.
        v = self.vector
        self.vector = v
        np.testing.assert_almost_equal(v, self.vector)

        # Get the dimension of the parameter space.
        ndim = len(v)

        # If a starting position is provided, ensure that the dimensions are
        # consistent.
        if start is not None:
            nwalkers = start.shape[0]
            if ndim != start.shape[1]:
                raise ValueError("Dimension mismatch: the dimension of the "
                                 "parameter space ({0}) doesn't ".format(ndim)
                                 + "match the dimension of the starting "
                                 "position ({0}).".format(start.shape[1]))
        else:
            nwalkers = kwargs.get("nwalkers", 16)

        # Parse the other input parameters.
        threads = kwargs.get("threads", 10)
        burnin = kwargs.get("burnin", [300, ])
        K = kwargs.get("K", 4)
        thin = kwargs.get("thin", 50)

        # Sanitize the data.
        self._prepare_data(*data)

        # Initialize a sampler.
        s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        # Do some HACKISH initialization. Start with a small ball and then
        # iterate (shrinking the size of the ball each time) until the range
        # of log-probabilities is "acceptable".
        ball = 1e-5
        p0 = emcee.utils.sample_ball(v, ball * v, size=nwalkers)
        lp = s._get_lnprob(p0)[0]
        dlp = np.var(lp)
        while dlp > 2:
            ball *= 0.5
            p0 = emcee.utils.sample_ball(v, ball * v, size=nwalkers)
            lp = s._get_lnprob(p0)[0]
            dlp = np.var(lp)

        # Run the burn-in iterations. After each burn-in run, cluster the
        # walkers and discard the worst ones.
        for i, nburn in enumerate(burnin):
            print(u"Burn-in pass {0}...".format(i + 1))
            p0, lprob, state = s.run_mcmc(p0, nburn, storechain=False)

            # Cluster the positions of the walkers at their final position
            # in log-probability using K-means.
            mix = mog.MixtureModel(K, np.atleast_2d(lprob).T)
            mix.run_kmeans()

            # Extract the cluster memberships.
            rs, rmxs = mix.kmeans_rs, np.argsort(mix.means.flatten())

            # Determine the "best" cluster that actually has walkers in it.
            for rmx in rmxs[::-1]:
                inds = rs == rmx
                good = p0[inds].T
                if np.shape(good)[1] > 0:
                    break

            # Compute the mean and covariance of the ensemble of good walkers.
            mu, cov = np.mean(good, axis=1), np.cov(good)

            # Re-sample the "bad" walkers from the Gaussian computed above.
            nbad = np.sum(~inds)
            if nbad == 0:
                print(u"  ... No walkers were rejected.")
                break

            p0[~inds] = np.random.multivariate_normal(mu, cov, nbad)

            # Hack to ensure that none of the re-sampled walkers fall outside
            # of the prior or have an infinite log-probability for other
            # reasons. NOTE: this could go on forever but that's pretty
            # unlikely :-).
            for n in np.arange(len(p0))[~inds]:
                lp = self.lnprob(p0[n])
                while np.isinf(lp):
                    p0[n] = np.random.multivariate_normal(mu, cov)
                    lp = self.lnprob(p0[n])

            print(u"  ... Rejected {0} walkers.".format(nbad))

            # Reset the chain to clear all the settings from burn-in.
            s.reset()

        # Get the full list of parameters in the correct order.
        pars = self.parameters + self.star.parameters
        for p in self.planets:
            pars += p.parameters
        par_list = np.array([(str(p.name), str(pickle.dumps(p, 0)))
                             for p in pars])

        # Initialize the results file.
        with h5py.File(filename, u"w") as f:
            # Save the dataset to the file.
            f.create_dataset(u"data", data=np.vstack(self._data))

            # Save the list of parameters and their pickled representations
            # to the file.
            f.create_dataset("parlist", data=par_list)

            # Add a group and headers for the MCMC results.
            g = f.create_group(u"mcmc")

            # ==================================================
            #
            # FIXME: ADD OTHER PARAMETERS SO THAT WE CAN PLOT
            #        RESULTS FROM THIS FILE ALONE.
            #
            # ==================================================

            g.attrs[u"threads"] = threads
            g.attrs[u"burnin"] = ", ".join([unicode(b) for b in burnin])
            g.attrs[u"iterations"] = iterations
            g.attrs[u"thin"] = thin

            # Create the datasets that will hold the MCMC results.
            c_ds = g.create_dataset(u"chain", (nwalkers, iterations, ndim),
                                    dtype=np.float64)
            lp_ds = g.create_dataset(u"lnprob", (nwalkers, iterations),
                                     dtype=np.float64)

        assert 0

        self._sampler = None

        if restart is None:
            pass

        # Reset and rerun.
        for i, (pos, lnprob, state) in enumerate(s.sample(p0,
                                                          thin=thin,
                                                          iterations=niter)):
            if i % thin == 0:
                print(i, np.mean(s.acceptance_fraction))
                with h5py.File(filename, u"a") as f:
                    g = f[u"mcmc"]
                    c_ds = g[u"chain"]
                    lp_ds = g[u"lnp"]

                    g.attrs[u"iterations"] = s.iterations
                    g.attrs[u"naccepted"] = s.naccepted
                    g.attrs[u"state"] = pickle.dumps(state)

                    ind = i0 + int(i / thin)
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        # Let's see some stats.
        print(u"Acceptance fraction: {0:.2f} %"
                .format(100 * np.mean(s.acceptance_fraction)))

        try:
            print(u"Autocorrelation time: {0}".format(
                    thin * s.acor))
        except RuntimeError:
            print(u"Autocorrelation time: too short")

        self._sampler = s
        return self._sampler.flatchain

    def plot_fit(self, true_ldp=None):
        p = Process(target=_async_plot, args=(u"_lc_and_ldp", self, true_ldp))
        p.start()
        self._processes.append(p)

    def _lc_and_ldp(self, true_ldp):
        time, flux, ivar = self._data
        chain = self._sampler.flatchain

        # Compute the best fit period.
        for i in range(self.nplanets):
            if u"a{0}".format(i) in self._pars:
                a = np.exp(np.median(chain[:,
                                self._pars.keys().index(u"a{0}".format(i))]))
            else:
                a = self.a[i]

            T = self.planets[i].get_period(self.mstar)

            # Generate light curve samples.
            t = np.linspace(0, T, 500)
            t2 = np.linspace(time.min(), time.max(),
                    int(100 * (time.max() - time.min() / T)))
            f = np.empty((len(chain), len(t)))
            f2 = np.zeros(len(t2))
            ld = [self.ldp.plot()[0],
                  np.empty((len(chain), 2 * len(self.ldp.bins)))]
            for ind, v in enumerate(chain):
                f[ind, :] = self.from_vector(v).lightcurve(t)
                f2 += self.lightcurve(t2)
                ld[1][ind, :] = self.ldp.plot()[1]
            f = f.T
            f2 = f2.T / len(chain)
            ld[1] = ld[1].T

            # Plot the fit.
            mu = np.median(flux)
            pl.figure()
            ax = pl.axes([0.15, 0.15, 0.8, 0.8])
            ax.plot(time % T, 1000 * (flux / mu - 1), u".k", alpha=0.5)
            ax.set_xlabel(u"Phase [days]")
            ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")
            pl.savefig(u"lc_{0}.png".format(i), dpi=300)

            ax.plot(t, 1000 * (f / mu - 1), u"#4682b4", alpha=0.04)
            pl.savefig(u"lc_fit_{0}.png".format(i), dpi=300)

            if i == 0:
                # Plot the full fit.
                pl.clf()
                ax = pl.axes([0.15, 0.15, 0.8, 0.8])
                ax.plot(time, 1000 * (flux / mu - 1), u".k", alpha=0.5)
                ax.set_xlabel(u"Time [days]")
                ax.set_ylabel(r"Relative Brightness Variation "
                              r"[$\times 10^{-3}$]")
                pl.savefig(u"lc_full.png", dpi=300)

                ax.plot(t2, 1000 * (f2 / mu - 1), u"#4682b4")
                pl.savefig(u"lc_full_fit.png", dpi=300)

        # Plot the limb-darkening.
        pl.clf()
        pl.plot(*ld, color=u"#4682b4", alpha=0.08)
        if true_ldp is not None:
            pl.plot(*true_ldp, color=u"k", lw=2)
        pl.savefig(u"ld.png", dpi=300)

    def plot_triangle(self, truths=None):
        p = Process(target=_async_plot, args=(u"_triangle", self, truths))
        p.start()
        self._processes.append(p)

    def _triangle(self, truths):
        assert self._data is not None and self._sampler is not None, \
                u"You have to fit some data first."

        chain = self._sampler.flatchain

        # Plot the parameter histograms.
        plotchain = self._sampler.flatchain
        inds = []
        for i, (k, p) in enumerate(self._pars.iteritems()):
            plotchain[:, i] = p.iconv(chain[:, i])
            if u"ldp" not in k:
                inds.append(i)

        labels = [str(p) for i, (k, p) in enumerate(self._pars.iteritems())
                            if i in inds]

        # Add the log-prob values too.
        lp = self._sampler.lnprobability.flatten()
        lp /= np.max(lp)
        plotchain = np.hstack([plotchain,
                    np.atleast_2d(lp).T])
        inds.append(plotchain.shape[1] - 1)
        labels.append(r"$\propto$ log-prob")

        if truths is not None:
            truths = [truths.get(k)
                    for i, (k, p) in enumerate(self._pars.iteritems())
                    if i in inds]

        triangle.corner(plotchain[:, inds], labels=labels, bins=20,
                                truths=truths)

        pl.savefig(u"triangle.png")


def _async_plot(pltype, ps, *args):
    return getattr(ps, pltype)(*args)


# class LDPParameter(LogParameter):

#     def getter(self, ps):
#         return self.conv(ps.ldp.intensity[self.ind]
#                          - ps.ldp.intensity[self.ind + 1])

#     def setter(self, ps, val):
#         j = self.ind
#         ps.ldp.intensity.__setitem__(j + 1,
#                                      ps.ldp.intensity[j] - self.iconv(val))
