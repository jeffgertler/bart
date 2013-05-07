#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various tools and constants for working with Kepler data.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["spline_detrend", "fiducial_ldp", "API", "EXPOSURE_TIMES",
           "TIME_ZERO"]

import os
import json
import requests
import numpy as np

try:
    from scipy.interpolate import LSQUnivariateSpline
    LSQUnivariateSpline = LSQUnivariateSpline
except ImportError:
    LSQUnivariateSpline = None

from .ldp import QuadraticLimbDarkening
from . import _bart


EXPOSURE_TIMES = [54.2, 1626.0]
TIME_ZERO = 2454833.0


def spline_detrend(x, y, yerr=None, Q=4, dt=4., tol=1.25e-3, maxiter=15,
                   fill_times=True, maxditer=4, nfill=4):
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
    if LSQUnivariateSpline is None:
        raise ImportError("scipy is required for spline de-trending.")

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
    if fill_times:
        inds = x[1:] - x[:-1] > 10 ** (-1.25)
        for i in np.arange(len(x))[inds]:
            t = _add_knots(t, x[i], x[i + 1], N=nfill)

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

        t = _add_knots(t, x[i], x[i + 1], N=np.max([nfill, 4]))

    return p


def _add_knots(t, t1, t2, N=3):
    """
    A hack for adding ``N`` samples in a given region and removing any other
    samples in that region.

    """
    return np.sort(np.append(t[(t < t1) + (t > t2)], np.linspace(t1, t2, N)))


class LDCoeffAdaptor(object):
    """
    Wrapper around various theoretical models for the coefficients of limb
    darkening profiles for Kepler stars.

    :param model: (optional)
        The name of the model that you would like to use. The currently
        supported models are ``sing09`` and ``claret11`` and the default
        is ``sing09`` because it is packaged with the standard Bart install.
        When you use ``claret11`` for the first time, it will download the
        data file and save it to ``~/.bart/ldcoeffs/claret11.txt``.

    """

    allowed_models = ["sing09", "claret11"]
    claret_url = "http://broiler.astrometry.net/~dfm265/ldcoeffs/claret.txt"

    def __init__(self, model="sing09"):
        if model not in self.allowed_models:
            raise TypeError(("Unrecognized model '{0}'. The allowed values "
                             "are {1}.").format(model, self.allowed_models))

        if model == "sing09":
            fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ldcoeffs", "sing09.txt")
            data = np.loadtxt(fn, skiprows=10, usecols=range(6))
            self.T = data[:, 0]
            self.logg = data[:, 1]
            self.feh = data[:, 2]
            self.mu1 = data[:, 4]
            self.mu2 = data[:, 5]

        else:
            fn = os.path.expanduser(os.path.join("~", ".bart", "ldcoeffs",
                                                 "claret11.txt"))
            if not os.path.exists(fn):
                print("Downloading the data file for the 'claret11` model")
                r = requests.get(self.claret_url)
                if r.status_code != requests.codes.ok:
                    r.raise_for_status()
                try:
                    os.makedirs(os.path.dirname(fn))
                except os.error:
                    pass
                open(fn, "w").write(r.content)
                print("  .. Finished.")

            data = np.loadtxt(fn, skiprows=59, delimiter="|",
                              usecols=range(7))
            self.T = data[:, 2]
            self.logg = data[:, 1]
            self.feh = data[:, 3]
            self.mu1 = data[:, 5]
            self.mu2 = data[:, 6]

    def get_coeffs(self, teff, logg=None, feh=None):
        """
        Get the coefficients of the quadratic limb darkening profile given by
        theory.

        :param teff:
            The effective temperature in degrees K.

        :param logg: (optional)
            The log10 surface gravity in cm/s/s.

        :param feh: (optional)
            The metallicity [Fe/H].

        """
        T0 = self.T[np.argmin(np.abs(self.T - teff))]
        inds = self.T == T0
        if logg is not None:
            lg0 = self.logg[np.argmin(np.abs(self.logg - logg))]
            inds *= self.logg == lg0
        if feh is not None:
            feh0 = self.feh[np.argmin(np.abs(self.feh - feh))]
            inds *= self.feh == feh0
        return np.mean(self.mu1[inds]), np.mean(self.mu2[inds])


def fiducial_ldp(teff=5778, logg=None, feh=None, model="sing09",
                 bins=None, alpha=1.0):
    """
    Get the standard Kepler limb-darkening profile.

    :param teff: (optional)
        The effective temperature in degrees K.

    :param logg: (optional)
        The log10 surface gravity in cm/s/s.

    :param feh: (optional)
        The metallicity [Fe/H].

    :model: (optional)
        The theoretical model to use. See :class:`LDCoeffAdaptor`.

    :param bins: (optional)
        Either the number of radial bins or a list of bin edges. If ``bins``
        is omitted, the functional form of the profile is returned.

    :param alpha: (optional)
        The power to scale the radial bin positions by.

    """
    # Find the LD coefficients from the theoretical models.
    a = LDCoeffAdaptor(model=model)
    mu1, mu2 = a.get_coeffs(teff, logg=logg, feh=feh)

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

        :param params:
            Any other search parameters.

        """
        params["action"] = params.get("action", "Search")
        params["outputformat"] = "JSON"
        params["coordformat"] = "dec"
        params["verb"] = 3
        r = requests.get(self.base_url.format(category), params=params)
        if r.status_code != requests.codes.ok:
            r.raise_for_status()

        try:
            data = r.json()
        except ValueError:
            return None

        # Fix the dumb-ass MAST field names.
        if category in _adaptors:
            result = []
            for d in data:
                tmp = {}
                for k, k2 in _adaptors[category].iteritems():
                    v = d[k]
                    if v is not None and v != "":
                        tmp[k2[0]] = k2[1](v)
                result.append(tmp)

        else:
            result = data

        return result

    def kois(self, **params):
        """
        Get a list of all the KOIs.

        """
        params["max_records"] = params.pop("max_records", 100)
        params["ordercolumn1"] = "kepoi"

        # Special case to deal with the ``kepoi<N`` type queries.
        if unicode(params.get("kepoi", " ")[0]) == "<":
            maxkoi = float(params["kepoi"][1:])
        else:
            maxkoi = 1e10

        # Submit the initial request.
        kois = self.request("koi", **params)
        if kois is None:
            raise StopIteration()

        # Yield each KOI as a generator.
        for k in kois:
            if float(k["kepoi"]) > maxkoi:
                raise StopIteration()
            yield k

        # Try to get more KOIs if they exist.
        while True:
            params["kepoi"] = ">{0}".format(kois[-1]["kepoi"])
            kois = self.request("koi", **params)
            if kois is None:
                raise StopIteration()

            # Yield these ones too.
            for k in kois:
                if float(k["kepoi"]) > maxkoi:
                    raise StopIteration()
                yield k

    def planets(self, **params):
        """
        Get a list of all the confirmed planets.

        """
        planets = self.request("confirmed_planets", **params)

        if planets is None:
            raise StopIteration()

        for p in planets:
            yield p

    def data(self, kepler_id):
        """
        Get the :class:`bart.kepler.DataList` of observations associated with
        a particular Kepler ID.

        :param kepler_id:
            The Kepler ID.

        """
        data_list = self.request("data_search", ktc_kepler_id=kepler_id)
        if data_list is None:
            return []
        return APIDataList(data_list)


class APIDataList(object):
    """
    A list of :class:`bart.kepler.Datasets`.

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


_adaptors = {
        "koi": {

            "Kepler ID": ("kepid", int),
            "KOI Name": ("kepoi_name", unicode),
            "KOI Number": ("kepoi", unicode),
            "Disposition": ("koi_disposition", unicode),
            "RA (J2000)": ("degree_ra", float),
            "Dec (J2000)": ("degree_dec", float),
            "Time of Transit Epoch": ("koi_time0bk", float),
            "Time err1": ("koi_time0bk_err1", float),
            "Time_err2": ("koi_time0bk_err2", float),
            "Period": ("koi_period", float),
            "Period err1": ("koi_period_err1", float),
            "Period err2": ("koi_period_err2", float),
            "Transit Depth": ("koi_depth", float),
            "Depth err1": ("koi_depth_err1", float),
            "Depth err2": ("koi_depth_err2", float),
            "Duration": ("koi_duration", float),
            "Duration err1": ("koi_duration_err1", float),
            "Duration err2": ("koi_duration_err2", float),
            "Ingress Duration": ("koi_ingress", float),
            "Ingress err1": ("koi_ingress_err1", float),
            "Ingress err2": ("koi_ingress_err2", float),
            "Impact Parameter": ("koi_impact", float),
            "Impact Parameter err1": ("koi_impact_err1", float),
            "Impact Parameter err2": ("koi_impact_err2", float),
            "Inclination": ("koi_incl", float),
            "Inclination err1": ("koi_incl_err1", float),
            "Inclination err2": ("koi_incl_err2", float),
            "Semi-major Axis": ("koi_sma", float),
            "Semi-major Axus err1": ("koi_sma_err1", float),
            "Semi-major Axis err2": ("koi_sma_err2", float),
            "Eccentricity": ("koi_eccen", float),
            "Eccentricity err1": ("koi_eccen_err1", float),
            "Eccentricity err2": ("koi_eccen_err2", float),
            "Long of Periastron": ("koi_longp", float),
            "Long err1": ("koi_longp_err1", float),
            "Long err2": ("koi_longp_err2", float),
            "r/R": ("koi_ror", float),
            "r/R err1": ("koi_ror_err1", float),
            "r/R err2": ("koi_ror_err2", float),
            "a/R": ("koi_dor", float),
            "a/R err1": ("koi_dor_err1", float),
            "a/R err2": ("koi_dor_err2", float),
            "Planet Radius": ("koi_prad", float),
            "Planet Radius err1": ("koi_prad_err1", float),
            "Planet Radius err2": ("koi_prad_err2", float),
            "Teq": ("koi_teq", float),
            "Teq err1": ("koi_teq_err1", float),
            "Teq err2": ("koi_teq_err2", float),
            "Teff": ("koi_steff", float),
            "Teff err1": ("koi_steff_err1", float),
            "Teff err2": ("koi_steff_err2", float),
            "log(g)": ("koi_slogg", float),
            "log(g) err1": ("koi_slogg_err1", float),
            "log(g) err2": ("koi_slogg_err2", float),
            "Metallicity": ("koi_smet", float),
            "Metallicity err1": ("koi_smet_err1", float),
            "Metallicity err2": ("koi_smet_err2", float),
            "Stellar Radius": ("koi_srad", float),
            "Stellar Radius err1": ("koi_srad_err1", float),
            "Stellar Radius err2": ("koi_srad_err2", float),
            "Stellar Mass": ("koi_smass", float),
            "Stellar Mass err2": ("koi_smass_err2", float),
            "Stellar Mass err1": ("koi_smass_err1", float),
            "Age": ("koi_sage", float),
            "Age err1": ("koi_sage_err1", float),
            "Age err2": ("koi_sage_err2", float),
            "Provenance": ("koi_sparprov", unicode),
            "Quarters": ("koi_quarters", unicode),
            "Limb Darkening Model": ("koi_limbdark_mod", unicode),
            "Limb Darkening Coeff1": ("koi_ldm_coeff1", float),
            "Limb Darkening Coeff2": ("koi_ldm_coeff2", float),
            "Limb Darkening Coeff3": ("koi_ldm_coeff3", float),
            "Limb Darkening Coeff4": ("koi_ldm_coeff4", float),
            "Transit Number": ("koi_num_transits", int),
            "Max single event sigma": ("koi_max_sngle_ev", float),
            "Max Multievent sigma": ("koi_max_mult_ev", float),
            "KOI count": ("koi_count", int),
            "Binary Discrimination": ("koi_bin_oedp_sig", float),
            "False Positive Bkgnd ID": ("koi_fp_bkgid", unicode),
            "J-band diff": ("koi_fp_djmag", unicode),
            "Comments": ("koi_comment", unicode),
            "Transit Model": ("koi_trans_mod", unicode),
            "Transit Model SNR": ("koi_model_snr", float),
            "Transit Model DOF": ("koi_model_dof", float),
            "Transit Model chisq": ("koi_model_chisq", float),
            "FWM motion signif.": ("koi_fwm_stat_sig", float),
            "gmag": ("koi_gmag", float),
            "gmag err": ("koi_gmag_err", float),
            "rmag": ("koi_rmag", float),
            "rmag err": ("koi_rmag_err", float),
            "imag": ("koi_imag", float),
            "imag err": ("koi_imag_err", float),
            "zmag": ("koi_zmag", float),
            "zmag err": ("koi_zmag_err", float),
            "Jmag": ("koi_jmag", float),
            "Jmag err": ("koi_jmag_err", float),
            "Hmag": ("koi_hmag", float),
            "Hmag err": ("koi_hmag_err", float),
            "Kmag": ("koi_kmag", float),
            "Kmag err": ("koi_kmag_err", float),
            "kepmag": ("koi_kepmag", float),
            "kepmag err": ("koi_kepmag_err", float),
            "Delivery Name": ("koi_delivname", unicode),
            "FWM SRA": ("koi_fwm_sra", float),
            "FWM SRA err": ("koi_fwm_sra_err", float),
            "FWM SDec": ("koi_fwm_sdec", float),
            "FWM SDec err": ("koi_fwm_sdec_err", float),
            "FWM SRAO": ("koi_fwm_srao", float),
            "FWM SRAO err": ("koi_fwm_srao_err", float),
            "FWM SDeco": ("koi_fwm_sdeco", float),
            "FWM SDeco err": ("koi_fwm_sdeco_err", float),
            "FWM PRAO": ("koi_fwm_prao", float),
            "FWM PRAO err": ("koi_fwm_prao_err", float),
            "FWM PDeco": ("koi_fwm_pdeco", float),
            "FWM PDeco err": ("koi_fwm_pdeco_err", float),
            "Dicco MRA": ("koi_dicco_mra", float),
            "Dicco MRA err": ("koi_dicco_mra_err", float),
            "Dicco MDec": ("koi_dicco_mdec", float),
            "Dicco MDec err": ("koi_dicco_mdec_err", float),
            "Dicco MSky": ("koi_dicco_msky", float),
            "Dicco MSky err": ("koi_dicco_msky_err", float),
            "Dicco FRA": ("koi_dicco_fra", float),
            "Dicco FRA err": ("koi_dicco_fra_err", float),
            "Dicco FDec": ("koi_dicco_fdec", float),
            "Dicco FDec err": ("koi_dicco_fdec_err", float),
            "Dicco FSky": ("koi_dicco_fsky", float),
            "Dicco FSky err": ("koi_dicco_fsky_err", float),
            "Dikco MRA": ("koi_dikco_mra", float),
            "Dikco MRA err": ("koi_dikco_mra_err", float),
            "Dikco MDec": ("koi_dikco_mdec", float),
            "Dikco MDec err": ("koi_dikco_mdec_err", float),
            "Dikco MSky": ("koi_dikco_msky", float),
            "Dikco MSky err": ("koi_dikco_msky_err", float),
            "Dikco FRA": ("koi_dikco_fra", float),
            "Dikco FRA err": ("koi_dikco_fra_err", float),
            "Dikco FDec": ("koi_dikco_fdec", float),
            "Dikco FDec err": ("koi_dikco_fdec_err", float),
            "Dikco FSky": ("koi_dikco_fsky", float),
            "Dikco FSky err": ("koi_dikco_fsky_err", float),
            "Last Update": ("rowupdate", unicode),

        },

        "confirmed_planets": {

            "Planet Name": ("kepler_name", unicode),
            "KOI": ("koi", unicode),
            "Kepler ID": ("kepid", int),
            "RA (J2000)": ("kic_ra", float),
            "Dec (J2000)": ("kic_dec", float),
            "Planet temp": ("eqt", float),
            "Planet Radius": ("prad", float),
            "Transit duration": ("duration", float),
            "Period": ("period", float),
            "Semi-major Axis": ("sma", float),
            "Metallicity": ("feh", float),
            "Stellar Mass": ("mass", float),
            "Stellar Radius": ("stellar_radius", float),
            "Stellar Teff": ("teff", float),
            "Logg": ("logg", float),
            "KEP Mag": ("kepmag", float),
            "KOI Name": ("kepoi_name", unicode),
            "Last Update": ("last_updated", unicode),

        },

    }
