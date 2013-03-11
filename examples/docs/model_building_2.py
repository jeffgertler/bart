#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl

from bart.parameters.base import LogParameter
from bart.results import Column

from model_building_1 import generate_synthetic_data


np.random.seed(123)


if __name__ == "__main__":
    kepler6, lc, sc = generate_synthetic_data()

    # Add fit parameters.
    kepler6.planets[0].parameters.append(LogParameter(r"$a$", "a"))
    kepler6.planets[0].parameters.append(LogParameter(r"$r$", "r"))

    # Add the datasets.
    kepler6.add_dataset(lc)
    kepler6.add_dataset(sc)
    print(len(lc), len(sc))

    kepler6.run_mcmc(500, thin=10)
    results = kepler6.results(thin=1, burnin=10)
    results.corner_plot([
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
        ])

    assert 0

    # The negative log-probability function.
    def nlp(p):
        ll = kepler6.lnprob(p)
        return -np.nan_to_num(ll)

    print(kepler6.vector)

    # The initial condition.
    p0 = kepler6.vector * (1 + 2e-3 * np.random.randn(len(kepler6.vector)))

    # Optimize.
    result = op.minimize(nlp, p0, method="L-BFGS-B")
    print(result)

    P = kepler6.planets[0].get_period(kepler6.star.mass)
    pl.plot((lc.time - 0.5 * P) % P + 0.5 * P, lc.flux, ".k")

    t = np.linspace(0.5 * P, 1.5 * P, 5000)
    t_fold = (t - 0.5 * P) % P + 0.5 * P
    pl.plot(t, kepler6.lightcurve(t), "r")
    pl.savefig("fit.png")
