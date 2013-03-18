#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import numpy as np
from numpy.testing import assert_allclose

from .. import _bart
from ..ldp import QuadraticLimbDarkening as QLD


def test_no_occultation():
    """
    Check the behavior in the absence of occultation.

    """
    area = _bart.occarea(1.0, 0.5, 2.0)
    assert_allclose(area, 0)

    area = _bart.occarea(1.0, 0.0, 0.5)
    assert_allclose(area, 0)

    area = _bart.occarea(1.0, 0.1, 1.1)
    assert_allclose(area, 0)


def test_full_occultation():
    """
    Check the behavior when the planet is fully in front the star.

    """
    p = 0.1
    area = _bart.occarea(1.0, p, 0.9)
    assert_allclose(area, np.pi * 0.1 ** 2)

    area = _bart.occarea(1.0, p, 0.0)
    assert_allclose(area, np.pi * 0.1 ** 2)

    area = _bart.occarea(1.0, 1.0, 0.0)
    assert_allclose(area, np.pi)


def test_mandel_agol():
    """
    Run a functional test against the output from the Mandel & Agol code.

    Note: this only approximately compares the results (to an absolute
    tolerance of 1.25e-6 because of the first-order histogram approximation.

    """
    # Read in the test data.
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "ld_test_data.dat")
    with open(fn) as f:
        lines = f.readlines()

    g1, g2, p = [float(c) for c in lines[0][2:].split()]
    b, lam0 = zip(*[l.split() for l in lines[1:]])
    b, lam0 = np.array(b, dtype=float), np.array(lam0, dtype=float)

    # Construct a limb darkening profile.
    bins = np.linspace(0, 1, 1000)[1:]
    ld = QLD(g1, g2).histogram(bins)

    # Compute the Bart approximation.
    lam = _bart.ldlc(p, ld.bins, ld.intensity, b)
    assert_allclose(lam, lam0, rtol=0, atol=1.25e-6)
