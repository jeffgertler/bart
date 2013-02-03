#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A BART example that can fit the light curve for any Kepler confirmed planets.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
from bart import kepler

import numpy as np
import matplotlib.pyplot as pl


def main(name="KEPLER-4 b"):
    # Resolve the planet name.
    api = kepler.API()
    planets = api.planets(kepler_name=name)
    if planets is None:
        print("No planet with the name '{0}'. Try: 'KEPLER-4 b'.".format(name))
        sys.exit(1)

    # Find all of the planets in the same system.
    kepid = planets[0]["Kepler ID"]
    planets = api.planets(kepid=kepid)
    if planets is None:
        print("Something went wrong.")
        sys.exit(2)

    # Fetch the data.
    bp = "data"
    data_files = api.data(kepid).fetch_all(basepath=bp)

    time, flux, ferr = np.array([]), np.array([]), np.array([])
    for fn in data_files:
        t, f, fe = kepler.load(fn)
        time = np.append(time, t)
        flux = np.append(flux, f)
        ferr = np.append(ferr, fe)

    pl.plot(time % float(planets[0]["Period"]),
            flux, ".", alpha=0.3)
    # pl.xlim(250, 350)
    pl.ylim(0.96, 1.02)
    pl.savefig("blah2.png")


if __name__ == "__main__":
    main()
