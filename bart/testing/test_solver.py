#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the second order Newton's method solver for Kepler's equation.

Note: this solver doesn't solve to machine precision. We're only going to
check that it's good to better than 1.25e-6.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose
from .. import _bart


solver_precision = 1.25e-6


def psi2wt(psi, e):
    return psi - e * np.sin(psi)


def test_solver_easy():
    """
    Test the non-linear equation solver in a few simple cases.

    """
    psi, info = _bart.wt2psi(0., 0.)
    assert info == 0
    assert_allclose(psi, 0.0)

    psi, info = _bart.wt2psi(0., 1.0)
    assert info == 0
    assert_allclose(psi, 0.0)


def test_solver_fengji():
    """
    Test the non-linear equation solver in one particularly hard case.

    """
    psi0 = 0.000001
    e = 1.0
    psi, info = _bart.wt2psi(psi2wt(psi0, e), e)
    assert info == 0
    assert_allclose(psi, psi0, rtol=0, atol=solver_precision)


def test_solver_ranges():
    """
    Try a grid of eccentricities and mean anomalies to scope the range of
    possibilities.

    """
    N = 5000
    psi0 = np.pi
    e0 = np.linspace(0.9999, 1.0, N)
    wt0 = psi2wt(psi0, e0)
    psi = [_bart.wt2psi(w, e)[0] for w, e in zip(wt0, e0)]
    assert_allclose(psi, psi0, rtol=0, atol=solver_precision)

    psi0 = 1e5 + np.linspace(0.0, 2 * np.pi, N)
    wt0 = psi2wt(psi0, e0)
    psi = [_bart.wt2psi(w, e)[0] for w, e in zip(wt0, e0)]
    assert_allclose(psi, psi0 % (2 * np.pi), rtol=0, atol=solver_precision)


def test_failures():
    """
    Test a few expected failures for unphysical systems.

    """
    psi, info = _bart.wt2psi(np.pi, -0.5)
    assert info != 0

    psi, info = _bart.wt2psi(np.pi, 1.5)
    assert info != 0
