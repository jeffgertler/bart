#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["fiducial_ldp", "API", "EXPOSURE_TIMES", "TIME_ZERO"]

import os
import json
import requests
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from .ldp import QuadraticLimbDarkening
from . import _bart


EXPOSURE_TIMES = [54.2, 1626.0]
TIME_ZERO = 2454833.0


def spline_detrend(x, y, yerr=None, Q=4, dt=3., tol=1.25e-3, maxiter=15,
                   maxditer=4, nfill=2):
    """
    Use iteratively re-weighted least squares to fit a spline to the base
    trend in a time series. This is especially useful (and specifically
    tuned) for de-trending Kepler light curves.

    :param x:
        The sampled times.

    :param y:
        The fluxes corresponding to the times in ``x``.

    :param yerr: (optional)
        The 1-sigma error bars on ``y``.

    :param Q: (optional)
        The parameter controlling the severity of the re-weighting.

    :param dt: (optional)
        The initial spacing between time control points.

    :param tol: (optional)
        The convergence criterion.

    :param maxiter: (optional)
        The maximum number of re-weighting iterations to run.

    :param maxditer: (optional)
        The maximum number of discontinuity search iterations to run.

    :param nfill: (optional)
        The number of knots to use to fill in the gaps.

    """
    if yerr is None:
        yerr = np.ones_like(y)

    inds = np.argsort(x)
    x, y, yerr = x[inds], y[inds], yerr[inds]
    ivar = 1. / yerr / yerr
    w = np.array(ivar)

    # Build the list of knot locations.
    N = (x[-1] - x[0]) / dt + 2
    t = np.linspace(x[0], x[-1], N)[1:-1]

    # Refine knot locations around break points.
    inds = x[1:] - x[:-1] > 10 ** (-1.25)
    for i in np.arange(len(x))[inds]:
        t = add_knots(t, x[i], x[i + 1], N=nfill)

    for j in range(maxditer):
        s0 = None
        for i in range(maxiter):
            # Fit the spline.
            extra_t = np.append(t, [x[0], x[-1]])
            x0 = np.append(x, extra_t)
            inds = np.argsort(x0)
            y0 = np.append(y, np.ones_like(extra_t))[inds]
            w0 = np.append(w, np.ones_like(extra_t))[inds]
            p = LSQUnivariateSpline(x0[inds], y0, t, k=3, w=w0)

            # Compute chi_i ^2.
            chi = (y - p(x)) / yerr
            chi2 = chi * chi

            # Check for convergence.
            sigma = np.median(chi2)
            if s0 is not None and np.abs(s0 - sigma) < tol:
                break
            s0 = sigma

            # Re compute weights.
            w = ivar * Q / (chi2 + Q)

        # Find any discontinuities.
        i = _bart.discontinuities(x, chi, 0.5 * dt, Q, 1.0)
        if i < 0:
            return p

        t = add_knots(t, x[i], x[i + 1], N=np.max([nfill, 4]))

    return p


def add_knots(t, t1, t2, N=3):
    return np.sort(np.append(t[(t < t1) + (t > t2)], np.linspace(t1, t2, N)))


def window_detrend(x, y, yerr=None, dt=2):
    for i in range(len(x)):
        y[i] /= np.median(y[np.abs(x - x[i]) < dt])
    return y


def fiducial_ldp(teff=5778, logg=4.44, feh=0.0, bins=None, alpha=1.0):
    """
    Get the standard Kepler limb-darkening profile.

    :param bins:
        Either the number of radial bins or a list of bin edges.

    """
    # Read in the limb darkening coefficient table.
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ld.txt")
    data = np.loadtxt(fn, skiprows=10)

    # Find the closest point in the table.
    T0 = data[np.argmin(np.abs(data[:, 0] - teff)), 0]
    logg0 = data[np.argmin(np.abs(data[:, 1] - logg)), 1]
    feh0 = data[np.argmin(np.abs(data[:, 2] - feh)), 2]
    ind = (data[:, 0] == T0) * (data[:, 1] == logg0) * (data[:, 2] == feh0)
    mu1, mu2 = data[ind, 4:6][0]

    # Generate a quadratic limb darkening profile.
    ldp = QuadraticLimbDarkening(mu1, mu2)

    if bins is None:
        return ldp

    # Build the list of bins.
    try:
        nbins = len(bins)
    except TypeError:
        nbins = int(bins)
        bins = np.linspace(0, 1, nbins + 1)[1:] ** alpha

    # Return the non-parametric approximation.
    return ldp.histogram(bins)


class API(object):
    """
    Interact with the Kepler MAST API.

    """

    base_url = "http://archive.stsci.edu/kepler/{0}/search.php"

    def request(self, category, **params):
        """
        Submit a request to the API and return the JSON response.

        :param category:
            The table that you want to search.

        :param **kwargs:
            Any other search parameters.

        """
        params["action"] = params.get("action", "Search")
        params["outputformat"] = "JSON"
        params["verb"] = 3
        r = requests.get(self.base_url.format(category), params=params)
        if r.status_code != requests.codes.ok:
            r.raise_for_status()

        try:
            return r.json()
        except ValueError:
            return None

    def kois(self, **params):
        """
        Get a list of all the KOIs.

        """
        return self.request("koi", **params)

    def planets(self, **params):
        """
        Get a list of all the confirmed planets.

        """
        return self.request("confirmed_planets", **params)

    def data(self, kepler_id):
        """
        Get the :class:`DataList` of observations associated with a particular
        Kepler ID.

        :param kepler_id:
            The Kepler ID.

        """
        data_list = self.request("data_search", ktc_kepler_id=kepler_id)
        if data_list is None:
            return []
        return APIDataList(data_list)


class APIDataList(object):
    """
    A list of :class:`Datasets`.

    """

    def __init__(self, datasets):
        self._datasets = [APIDataset(d) for d in datasets]

    def __getitem__(self, i):
        return self._datasets[i]

    def __str__(self):
        return "[\n" + ",\n".join([unicode(d) for d in self._datasets]) + "\n]"

    def __repr__(self):
        return unicode(self)

    def fetch_all(self, basepath="."):
        try:
            os.makedirs(basepath)
        except os.error:
            pass

        results = [d.fetch(basepath) for d in self._datasets]
        return [r for r in results if r is not None]


class APIDataset(object):
    """
    A Kepler dataset.

    """

    data_url = "http://archive.stsci.edu/pub/kepler/lightcurves/{0}/{1}/{2}"
    fn_fmt = "{0}_{1}.fits"

    def __init__(self, spec):
        self._spec = spec

    def __getitem__(self, k):
        return self._spec[k]

    def __str__(self):
        return json.dumps(self._spec, indent=4)

    def __repr__(self):
        return unicode(self)

    def filename(self):
        suffix = "llc" if self["Target Type"] == "LC" else "slc"
        fn = self.fn_fmt.format(self["Dataset Name"], suffix).lower()
        return fn

    def url(self):
        kid = "{0:09d}".format(int(self["Kepler ID"]))
        url = self.data_url.format(kid[:4], kid, self.filename())
        return url

    def fetch(self, basepath, clobber=False):
        url = self.url()
        local_fn = os.path.join(basepath, self.filename())
        if os.path.exists(local_fn) and not clobber:
            return local_fn

        # Fetch the file.
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            return None
        open(local_fn, "wb").write(r.content)

        return local_fn
