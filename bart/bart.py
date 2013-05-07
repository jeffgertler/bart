#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["Star", "Planet", "PlanetarySystem"]

import os
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

from bart import _bart
from bart.km import km1d
from bart.ldp import LimbDarkening


_G = 2945.4625385377644


class Model(object):

    def __init__(self):
        self.parameters = []
        
        # Diagnostic code
        self.liketrace = np.array([])


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

    @property
    def spec(self):
        return np.array([self.mass, self.radius, self.flux])

    @spec.setter  # NOQA
    def spec(self, v):
        self.mass, self.radius, self.flux = v

    def get_semimajor(self, T):
        """
        Get the semi-major axis for a massless planet orbiting this star with
        the period ``T``.

        """
        return (_G * T * T * self.mass / (4 * np.pi * np.pi)) ** (1. / 3)


class Planet(Model):
    """
    A :class:`Planet` object represents the set of parameters describing a
    single planet in a planetary system.

    :param r:
        The size of the planet in Solar radii.

    :param a:
        The semi-major axis of the orbit in Solar radii.

    :param mass: (optional)
        The mass of the planet in Solar masses.

    :param t0: (optional)
        The time of a reference transit in days (the zero-point doesn't
        matter as long as you're consistent).

    :param e: (optional)
        The eccentricity of the orbit.

    :param pomega: (optional)
        The angle (in radians) of the orbital ellipse in the reference plane.

    :param ix: (optional)
        The inclination of the orbit around the perpendicular axis to the
        observer's line-of-sight in degrees.

    :param iy: (optional)
        The inclination of the orbit around the observes line-of-sight in
        degrees.

    """

    def __init__(self, r, a, mass=0.0, t0=0.0, e=0.0, pomega=0.0, ix=0.0,
                 iy=0.0):
        super(Planet, self).__init__()

        self.r = r
        self.a = a
        self.mass = mass
        self.t0 = t0
        self.e = e
        self.pomega = pomega
        self.ix = ix
        self.iy = iy

    @property
    def spec(self):
        return np.array([float(self.r), float(self.a), float(self.mass),
                         float(self.t0), float(self.e), float(self.pomega),
                         float(self.ix), float(self.iy)])

    @spec.setter  # NOQA
    def spec(self, v):
        (self.r, self.a, self.mass, self.t0, self.e, self.pomega, self.ix,
         self.iy) = v

    def get_mstar(self, P):
        """
        Get the mass of the host star implied by the semi-major axis of this
        planet and an input period.

        """
        a = self.a
        return a * a * a * 4 * np.pi * np.pi / _G / P / P - self.mass

    def get_period(self, mstar):
        """
        Get the period of this planet orbiting a star with a given mass.

        """
        a = self.a
        return 2 * np.pi * np.sqrt(a * a * a / _G / (mstar + self.mass))


class PlanetarySystem(Model):
    """
    A system of planets orbiting a star.

    :param star:
        A :class:`Star` object.

    :param iobs: (optional)
        The viewing angle in degrees. 90 is defined as edge on.

    """

    def __init__(self, star, basepath=".", iobs=90.0, rv0=0.0, pbad=0.0,
                 vbad=0.0):
        super(PlanetarySystem, self).__init__()

        self.basepath = basepath
        try:
            os.makedirs(basepath)
        except os.error:
            pass

        # Data.
        self.datasets = []

        # Priors.
        self.priors = []

        # The properties of the system as a whole.
        self.star = star
        self.iobs = iobs
        self.rv0 = rv0

        # The planets.
        self.planets = []

        # The background model.
        self.pbad = pbad
        self.vbad = vbad

    def add_dataset(self, ds):
        self.datasets.append(ds)

    def add_prior(self, p):
        self.priors.append(p)

    @property
    def spec(self):
        return np.concatenate([[float(self.iobs), float(self.rv0),
                                float(self.pbad), float(self.vbad)],
                                self.star.spec,
                              np.concatenate([p.spec for p in self.planets])])

    @spec.setter  # NOQA
    def spec(self, v):
        nspec = 4
        self.iobs = v[0]
        self.rv0 = v[1]
        self.pbad = v[2]
        self.vbad = v[3]

        ls = len(self.star.spec)
        self.star.spec = v[nspec:ls + nspec]
        lp = len(self.planets[0].spec)
        for i, p in enumerate(self.planets):
            p.spec = v[ls + nspec + i * lp:ls + nspec + (i + 1) * lp]

    def results(self, *args, **kwargs):
        from bart.results import ResultsProcess
        kwargs["basepath"] = kwargs.pop("basepath", self.basepath)
        return ResultsProcess(*args, **kwargs)

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
                              + [p.vector for p in self.planets]
                              + [d.vector for d in self.datasets])

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
        for d in self.datasets:
            d.vector = v[i:i + len(d)]
            i += len(d)

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

        for d in self.datasets:
            pv = d.sample(size, std=std)
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
            self.poissonlike()
            lnl = self.lnlike()
            if np.isinf(lnl) or np.isnan(lnl):
                return -np.inf

        except (FloatingPointError, AssertionError):
            return -np.inf
        
        # Diagnostic code
        np.append(self.liketrace, [lnl+lnp])
        
        return lnl + lnp

    def lnprior(self):
        """
        Compute the log-prior of the current model.

        """
        lnp = [p(self) for p in self.priors]
        lnp += [p.lnprior(self) for p in self.parameters]
        lnp += [p.lnprior(self.star) for p in self.star.parameters]
        for planet in self.planets:
            lnp += [p.lnprior(planet) for p in planet.parameters]
        for d in self.datasets:
            lnp += [p.lnprior(d) for p in d.parameters]
        if np.isinf(np.any(lnp)):
            return -np.inf
        return np.sum(lnp)

    def poissonlike(self, dt=.01):
        """
        Compute the probabiity of the lightcurve based on photon data

        """
        photonrates = self.lightcurve(self.times)
        bintimes = np.arange(self.times[0], self.times[-1], dt)
        binrates = self.lightcurve(bintimes)
        prob = np.sum(np.log(photonrates)) - dt * np.sum(binrates)
        #print(prob)
        return prob
        

    def lnlike(self):
        """
        Compute the ln-likelihood of the current model.

        """
        lnlike = 0.0

        N = np.sum([len(ds) for ds in self.datasets])

        for ds in self.datasets:
            # Add in the jitter.
            ivar = ds.ivar
            inds = ivar > 0
            ivar[inds] = 1. / (1. / ivar[inds] + ds.jitter * ds.jitter)

            if ds.__type__ == "lc":
                # Compute the "foreground" model probability.
                model = self.lightcurve(ds.time, texp=ds.texp)
                delta = ds.flux - ds.zp * model
                # e1 = np.log(1 - self.pbad) + 0.5 * np.log(ivar) \
                #      - 0.5 * delta * delta * ivar

                # # Compute the background model probability.
                # ivar_bg = np.zeros_like(ivar)
                # ivar_bg[inds] = 1. / (1. / ds.ivar[inds]
                #                       + self.vbad * self.vbad)
                # delta_bg = ds.flux - ds.zp
                # e2 = np.log(self.pbad) + 0.5 * np.log(ivar_bg) \
                #      - 0.5 * delta_bg * delta_bg * ivar_bg

                # lnlike += np.sum(np.logaddexp(e1, e2))
                lnlike += np.sum(0.5 * np.log(ivar) / N
                                 - 0.5 * delta * delta * ivar / N)

            elif ds.__type__ == "rv":
                model = self.radial_velocity(ds.time)
                delta = ds.rv - model
                c2 = np.sum(delta * delta * ivar) - np.sum(np.log(ivar))
                lnlike -= 0.5 * c2

            else:
                raise TypeError()

        return lnlike

    def _get_pars(self):
        r = [(p.mass, p.r, p.a, p.t0, p.e, p.pomega, p.ix, p.iy)
                                                    for p in self.planets]
        return zip(*r)

    def null_bgfunc(t):
        """
        | Generates a zeroed background 
        | FIXME: MAGIC NUMBER .001
        """
        return np.zeros_like(t)+.001
    
    def null_sfunc(t):
        """
        | Generates a perfect sensitivity (all ones)
        """
        return np.ones_like(t)

    def lightcurve(self, t, texp=0, K=1, bgfunc=null_bgfunc, sfunc=null_sfunc):
        """
        Get the light curve of the model at the current model.

        :param t:
            The times where the light curve should be evaluated.

        :param texp:
            The exposure time in seconds.

        :param K:
            The number of bins to use when integrating over exposure time.

        """
        mass, r, a, t0, e, pomega, ix, iy = self._get_pars()
        s = self.star
        ldp = s.ldp
        lc, info = _bart.lightcurve(t, texp / 68400., K, s.flux, s.mass,
                                s.radius, self.iobs,
                                mass, r, a, t0, e, pomega, ix, iy,
                                ldp.bins, ldp.intensity)
        assert info == 0, "Orbit computation failed. {0}".format(e)
        return lc * sfunc(t) + bgfunc(t)

    def radial_velocity(self, t):
        s = self.star
        result = np.zeros_like(t)
        for p in self.planets:
            pos, rv, info = _bart.solve_orbit(t, s.mass, p.mass, p.e, p.a,
                                            p.t0, p.pomega,
                                            np.radians(90 - self.iobs + p.ix),
                                            np.radians(p.iy), 1)
            result += rv

        # Add systemic velocity and convert to m/s.
        return self.rv0 + 8050.0 * result

    def optimize(self):
        objective = lambda p: -self(p)
        result = op.minimize(objective, self.vector)
        self.vector = result.x
        return result.x

    def get_photons(self, t):
        """
        generate a Poisson photon list from a light curve
        - fails when `self.lightcurve()` hits zero
        - requires user to input a sensible (ie, fine) time grid `t`
        - time grid t must be monotonically increasing
        """
        print("Generating photons")
        lc = self.lightcurve(t)
        times = np.empty(0)
        for i in range(len(t)-1):
            tbin = t[i+1] - t[i]
            times = np.append(times, t[i]+tbin*np.random.uniform(size = np.random.poisson(lam = tbin*lc[i])))
        self.t = t
        self.times = times
        print("Photons finished")
        pl.clf()
        pl.plot(t, lc)
        pl.hist(times, bins = t.max()-t.min())
        pl.savefig("/home/jmg789/bart/document/figures/lightcurve.png")
        print("Finished lightcurve.png")
        return self.times


    def run_mcmc(self, iterations, start=None, filename="mcmc.h5", **kwargs):
        """
        Fit the data using MCMC to get constraints on the parameters.

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
        outfn = os.path.join(self.basepath, filename)

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
        burnin = kwargs.get("burnin", [])
        K = kwargs.get("K", 4)
        thin = kwargs.get("thin", 1)

        # Initialize a sampler.
        s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        # Do some HACKISH initialization. Start with a small ball and then
        # iterate (shrinking the size of the ball each time) until the range
        # of log-probabilities is "acceptable".
        if start is None:
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
        else:
            p0 = np.array(start)

        # Run the burn-in iterations. After each burn-in run, cluster the
        # walkers and discard the worst ones.
        for i, nburn in enumerate(burnin):
            print(u"Burn-in pass {0}...".format(i + 1))
            p0, lprob, state = s.run_mcmc(p0, nburn)

            lp = s.lnprobability
            mu, rs = km1d(lp.reshape(lp.size), k=K)
            rs = rs.reshape(lp.shape)[:, -1]
            rmxs = np.argsort(mu)

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
        for d in self.datasets:
            pars += d.parameters
        par_list = np.array([str(pickle.dumps(p, 0)) for p in pars])

        # Pickle the initial conditions of ``self`` so that we can start again
        # from the same place.
        initial_spec = str(pickle.dumps(self, 0))

        # Import the current version number.
        from bart import __version__

        # Initialize the results file.
        with h5py.File(outfn, u"w") as f:
            # Keep track of the current Bart version.
            f.attrs["version"] = __version__

            # Save the list of parameters and their pickled representations
            # to the file.
            f.create_dataset("parlist", data=par_list)

            # Save the pickled version of ``self``.
            f.create_dataset("initial_pickle", data=initial_spec)

            f.create_dataset("datasets",
                             data=str(pickle.dumps(self.datasets)))

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
                with h5py.File(outfn, "a") as f:
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
