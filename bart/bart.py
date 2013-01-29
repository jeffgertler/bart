#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["Star", "Planet", "PlanetarySystem"]

import cPickle as pickle

import numpy as np
import emcee

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
            return np.concatenate([np.atleast_1d(p.conv(p.getter(self)))
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
            p.setter(self, p.iconv(v[i:i + len(p)]))
            i += len(p)

    def sample(self, size, std=1e-5):
        try:
            return np.concatenate([
                        np.atleast_2d(p.sample(self, std=std, size=size))
                                                for p in self.parameters
                                                if len(p) > 0], axis=0)
        except ValueError:
            return None

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

    def sample(self, size, std=1e-5):
        """

        """
        l = []
        v = super(PlanetarySystem, self).sample(size, std=std)
        if v is not None:
            l.append(v)

        sv = self.star.sample(size, std=std)
        if sv is not None:
            l.append(sv)

        for p in self.planets:
            pv = p.sample(size, std=std)
            if pv is not None:
                l.append(pv)

        return np.concatenate(l).T

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
        lnp = [p.lnprior(self) for p in self.parameters]
        lnp += [p.lnprior(self.star) for p in self.star.parameters]
        for planet in self.planets:
            lnp += [p.lnprior(planet) for p in planet.parameters]
        if np.isinf(np.any(lnp)):
            return -np.inf
        return np.sum(lnp)

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
        lc, info = _bart.lightcurve(t, s.flux, s.mass, s.radius, self.iobs,
                                r, a, t0, e, pomega, ix, iy,
                                ldp.bins, ldp.intensity)
        assert info == 0, "Orbit computation failed. {0}".format(e)
        return lc

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
        # Reset for the sake of pickling.
        self._sampler = None

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
        thin = kwargs.get("thin", 1)

        # Sanitize the data.
        self._prepare_data(*data)

        # Initialize a sampler.
        s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        # Do some HACKISH initialization. Start with a small ball and then
        # iterate (shrinking the size of the ball each time) until the range
        # of log-probabilities is "acceptable".
        print("Initializing walkers.")
        ball = 1e-5
        p0 = self.sample(nwalkers, std=ball)
        lp = s._get_lnprob(p0)[0]
        dlp = np.var(lp)
        while dlp > 2:
            ball *= 0.5
            p0 = self.sample(nwalkers, std=ball)
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
                s.reset()
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
        par_list = np.array([str(pickle.dumps(p, 0)) for p in pars])

        # Pickle the initial conditions of ``self`` so that we can start again
        # from the same place.
        initial_spec = str(pickle.dumps(self, 0))

        # Import the current version number.
        from bart import __version__

        # Initialize the results file.
        with h5py.File(filename, u"w") as f:
            # Keep track of the current Bart version.
            f.attrs["version"] = __version__

            # Save the dataset to the file.
            f.create_dataset("data", data=np.vstack(self._data))

            # Save the list of parameters and their pickled representations
            # to the file.
            f.create_dataset("parlist", data=par_list)

            # Save the pickled version of ``self``.
            f.create_dataset("initial_pickle", data=initial_spec)

            # Add a group and headers for the MCMC results.
            g = f.create_group("mcmc")

            # Save the MCMC meta data.
            g.attrs["threads"] = threads
            g.attrs["burnin"] = ", ".join([unicode(b) for b in burnin])
            g.attrs["iterations"] = iterations
            g.attrs["thin"] = thin

            # Create the datasets that will hold the MCMC results.
            N = int(iterations / thin)
            c_ds = g.create_dataset("chain", (nwalkers, N, ndim),
                                    dtype=np.float64)
            lp_ds = g.create_dataset("lnprob", (nwalkers, N),
                                     dtype=np.float64)

        # if restart is None:
        #     pass

        # Finally, run the production run.
        print("Running...")
        print("{0:8s}    {1}".format("Iter.", "Acc. Fraction"))
        status_fmt = "{0:8d}    {1:.2f}"
        for i, (pos, lnprob, state) in enumerate(s.sample(p0, thin=thin,
                                                    iterations=iterations)):
            if i % thin == 0:
                print(status_fmt.format(i, np.mean(s.acceptance_fraction)))
                with h5py.File(filename, "a") as f:
                    g = f["mcmc"]
                    c_ds = g["chain"]
                    lp_ds = g["lnprob"]

                    g.attrs["iterations"] = s.iterations
                    g.attrs["naccepted"] = s.naccepted
                    g.attrs["state"] = pickle.dumps(state)

                    ind = int(i / thin)
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        # Let's see some stats.
        print("Acceptance fraction: {0:.2f} %"
                .format(100 * np.mean(s.acceptance_fraction)))

        try:
            print("Autocorrelation time: {0}".format(
                    thin * s.acor))
        except RuntimeError:
            print("Autocorrelation time: too short")

        self._sampler = s
        return self._sampler.flatchain


# class LDPParameter(LogParameter):

#     def getter(self, ps):
#         return self.conv(ps.ldp.intensity[self.ind]
#                          - ps.ldp.intensity[self.ind + 1])

#     def setter(self, ps, val):
#         j = self.ind
#         ps.ldp.intensity.__setitem__(j + 1,
#                                      ps.ldp.intensity[j] - self.iconv(val))
