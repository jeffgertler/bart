#!/usr/bin/env python


import os
import sys

import numpy as np
import matplotlib.pyplot as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
from bart import BART


if __name__ == "__main__":
    ps = BART(1.0, 100.0, 0.0, ldp=[500, 0.39, 0.1], ldptype="quad")
    ps.add_planet(1.0, 5.2, 0.05, 4332.59, np.pi, 0.0)

    N = 50000
    t = 10000. * np.random.rand(N)
    ferr = 0.1 * np.random.rand(N)
    f = ps.lightcurve(t) + ferr * np.random.randn(N)

    pl.errorbar(t % 4332.59, f, yerr=ferr, fmt=".k")
    pl.xlim(2160, 2170)
    pl.savefig("solar_system.png")
