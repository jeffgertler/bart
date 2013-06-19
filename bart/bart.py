#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Star", "Planet", "PlanetarySystem"]

import numpy as np
from . import _bart
from .ld import LimbDarkening


_G = 2945.4625385377644


class Star(object):

    def __init__(self, mass=1.0, radius=1.0, flux=1.0, ldp=None):
        self.mass = mass
        self.radius = radius
        self.flux = flux

        # The limb darkening profile.
        if ldp is None:
            # Default to a uniformly illuminated star.
            self.ldp = LimbDarkening(1.0, 1.0)
        else:
            self.ldp = ldp

    def get_semimajor(self, T, planet_mass=0.0):
        """
        Get the semi-major axis for a massless planet orbiting this star with
        the period ``T``.

        """
        return (_G * T * T * (self.mass + planet_mass)
                / (4 * np.pi * np.pi)) ** (1. / 3)


class Planet(object):

    def __init__(self, r, a, mass=0.0, t0=0.0, e=0.0, pomega=0.0, ix=0.0,
                 iy=0.0):
        self.r = r
        self.a = a
        self.mass = mass
        self.t0 = t0
        self.e = e
        self.pomega = pomega
        self.ix = ix
        self.iy = iy

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


class PlanetarySystem(object):

    def __init__(self, star, iobs=90.0):
        self.star = star
        self.iobs = iobs
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

    def _get_pars(self):
        r = [(p.mass, p.r, p.a, p.t0, p.e, p.pomega, p.ix, p.iy)
             for p in self.planets]
        return zip(*r)

    def lightcurve(self, t, texp=1626.0, K=5):
        """
        Get the light curve of the model at the current model.

        :param t:
            The times where the light curve should be evaluated.

        :param texp:
            The exposure time in seconds.

        :param K:
            The number of bins to use when integrating over exposure time.

        """
        mass, r, a, t0, e, pomega, ix, iy = map(np.atleast_1d,
                                                self._get_pars())
        s = self.star
        ldp = s.ldp
        lc = _bart.lightcurve(np.atleast_1d(t), texp / 86400., K, s.flux,
                              s.mass, s.radius, self.iobs,
                              mass, r, a, t0, e, pomega, ix, iy,
                              np.atleast_1d(ldp.bins),
                              np.atleast_1d(ldp.intensity))
        return lc
