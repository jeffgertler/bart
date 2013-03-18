#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose
from .. import _bart


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

    """
    pass
