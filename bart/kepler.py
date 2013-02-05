#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["load", "fiducial_ldp"]

import os
import json
import pyfits
import requests
import numpy as np

from .ldp import QuadraticLimbDarkening


def load(fn):
    """
    Load, normalize and sanitize a Kepler light curve downloaded from MAST.
    The median flux is divided out of the flux and flux uncertainty for
    numerical stability.

    :param fn:
        The path to the FITS file.

    :return time:
        The times of the samples.

    :return flux:
        The brightness at the time samples.

    :return ferr:
        The quoted uncertainty on the flux.

    """
    f = pyfits.open(fn)
    lc = np.array(f[1].data)
    f.close()

    time = lc["TIME"]
    flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    # t0 = int(np.median(time[~np.isnan(time)]))
    # time = time - t0

    inds = ~np.isnan(time) * ~np.isnan(flux) * ~np.isnan(ferr)
    time, flux, ferr = time[inds], flux[inds], ferr[inds]

    mu = np.median(flux)
    flux /= mu
    ferr /= mu

    return (time, flux, ferr)


def fiducial_ldp(teff, logg, feh, bins=100, alpha=1.0):
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
    print("LDP Parameters", mu1, mu2)

    # Build the list of bins.
    try:
        nbins = len(bins)
    except TypeError:
        nbins = int(bins)
        bins = np.linspace(0, 1, nbins + 1)[1:] ** alpha

    # Generate a quadratic limb darkening profile.
    ldp = QuadraticLimbDarkening(mu1, mu2)

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
        return DataList(data_list)


class DataList(object):
    """
    A list of :class:`Datasets`.

    """

    def __init__(self, datasets):
        self._datasets = [Dataset(d) for d in datasets]

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

        return [d.fetch(basepath) for d in self._datasets]


class Dataset(object):
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
            r.raise_for_status()
        open(local_fn, "wb").write(r.content)

        return local_fn
