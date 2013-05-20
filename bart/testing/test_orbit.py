#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose
from .. import _bart


def test_eccentricity():
    """
    Check the effects of eccentricity on the orbits.

    """
    # System parameters.
    Ms = 1.0
    Mp = 0.0
    e = 1.0
    a = 10.0
    t0 = 0.0
    pomega = 0.0
    ix, iy = 0.0, 0.0

    # For an eccentricity of 1, the solve should fail at ``t = 0``.
    pos, rv, info = _bart.solve_orbit(0, Ms, Mp, e, a, t0, pomega, ix, iy,
                                      False)
    assert info != 0

    pos, rv, info = _bart.solve_orbit(0.5, Ms, Mp, e, a, t0, pomega, ix, iy,
                                      False)
    assert info != 0


def test_inclinations():
    """
    Test that the inclination parameters correctly rotate the orbit.

    """
    # System parameters.
    Ms = 1.0
    Mp = 0.0
    e = 0.0
    a = 10.0
    t0 = 0.0
    pomega = 0.0
    ix, iy = 0.0, 0.0

    t = np.linspace(0, 100, 200)

    # Without inclination, the orbit should be in the x-y plane.
    pos, rv, info = _bart.solve_orbit(t, Ms, Mp, e, a, t0, pomega, ix, iy,
                                      False)
    assert_allclose(pos[2, :], 0)

    # The ``ix`` parameter should rotate around y-axis.
    pos = [_bart.solve_orbit(0, Ms, Mp, e, a, t0, pomega, i, iy, False)[0][1]
           for i in np.linspace(0, np.pi, 10)]
    assert_allclose(pos, 0)

    # The ``iy`` parameter should rotate around x-axis (direction to the
    # observer).
    pos = [_bart.solve_orbit(0, Ms, Mp, e, a, t0, pomega, ix, iy, False)[0][0]
           for i in np.linspace(0, np.pi, 10)]
    assert_allclose(pos, a)


def test_positions():
    """
    Make sure that the orbit solves right at negative times.

    """
    # System parameters.
    Ms = 1.0
    Mp = 0.0
    e = 0.0
    a = 10.0
    t0 = 0.5
    pomega = 0.0
    ix, iy = 0.0, 0.0

    # For an eccentricity of 1, the solve should fail at ``t = 0``.
    pos, rv, info = _bart.solve_orbit([0., t0, 1.], Ms, Mp, e, a, t0,
                                      pomega, ix, iy, False)
    assert info == 0
    assert np.any(pos[:, 0] != pos[:, 1])
    assert pos[1, 0] == -pos[1, 2]
    assert pos[0, 0] == pos[0, 2]
