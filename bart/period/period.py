from __future__ import print_function


__all__ = [u"lc_model", u"chi2", u"find_period"]


from multiprocessing import Pool

import numpy as np
import scipy.optimize as op

import _periodogram

_default_order = 12


def lc_model(omega, amplitudes, order):
    a = amplitudes
    return lambda t: a[0] + np.sum([a[2 * i + 1] *
                                        np.sin((i + 1) * omega * t)
                                    + a[2 * i + 2] *
                                        np.cos((i + 1) * omega * t)
                                            for i in range(order)], axis=0)


def chi2(model, time, flux, ferr):
    if ferr is None:
        ferr = np.ones_like(flux)
    return np.sum((flux - model(time)) ** 2 / ferr / ferr)


class _fit_wrapper(object):
    def __init__(self, time, flux, ferr, order):
        self.time = time
        self.flux = flux
        self.ferr = ferr
        self.order = order

    def calc(self, omega, t, f, ferr):
        a, chi2, info = _periodogram.get_chi2(omega, t, f, ferr, self.order)

        if info != 0:
            print(u"Fit failed")
            return np.inf

        # Calculate the amplitudes and make sure that the 1st order
        # dominates.
        amp = a[1::2] ** 2 + a[2::2] ** 2
        if a[0] < 0 or np.any(amp[0] < amp[1:]):
            return np.inf

        return chi2

    def __call__(self, omega):
        # Check to make sure that the period isn't _very_ close to a day.
        # if np.abs(omega - 2 * np.pi) < 0.05:
        #     return 1e10

        # Do the fit.
        if not isinstance(self.time, dict):
            return self.calc(omega, self.time, self.flux, self.ferr)

        r = 0.0
        for k in self.time:
            tmp = self.calc(omega, self.time[k], self.flux[k], self.ferr[k])
            if np.isinf(tmp):
                return 1e10
            r += tmp

        return r


class _op_wrapper(object):
    def __init__(self, time, flux, ferr, order):
        self.time = time
        self.flux = flux
        self.ferr = ferr
        self.order = order

    def chi(self, w):
        return sum([_periodogram.get_chi(w, self.time[k], self.flux[k],
                self.ferr[k], self.order)[1] for k in self.time])

    def __call__(self, omega):
        res = op.leastsq(self.chi, omega, full_output=True)
        return res[0], np.sum(res[2][u"fvec"] ** 2)


def find_period(time, flux, ferr=None, order=None, N=30, Ts=[0.2, 1.3],
        pool=None):
    """
    Find the best fit period of a lightcurve by doing a grid
    search then a non-linear refinement step.

    """
    if order is None:
        order = _default_order

    # Set up a grid to do a grid search in frequency space.
    domega = 0.2 / min([time[k].max() - time[k].min() for k in time])
    omegas = 2 * np.pi * np.arange(1 / max(Ts), 1 / min(Ts), domega)

    # Do a parallel grid search.
    if pool is None:
        pool = Pool()
    chi2 = pool.map(_fit_wrapper(time, flux, ferr, order), omegas)

    # Sort the results by chi2.
    inds = np.argsort(chi2)

    # Refine the top `N` best fits.
    ref = pool.map(_op_wrapper(time, flux, ferr, order), omegas[inds[:N]])

    # Clean up... otherwise we get the error: _Too many open files_.
    pool.close()
    pool.join()
    del pool

    # Sort the refinements and return the best.
    omega = min(ref, key=lambda x: x[1])
    return 2 * np.pi / float(omega[0])
