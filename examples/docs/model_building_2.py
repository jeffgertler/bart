#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np

import bart.parameters as pars
from bart.results import ResultsProcess, Column

from model_building_1 import generate_synthetic_data


np.random.seed(123)


if __name__ == "__main__":
    kepler6, lc, sc = generate_synthetic_data()

    # Add fit parameters.
    kepler6.planets[0].parameters.append(pars.LogParameter(r"$\ln\,a$", "a"))
    kepler6.planets[0].parameters.append(pars.LogParameter(r"$\ln\,r$", "r"))
    kepler6.planets[0].parameters.append(pars.Parameter(r"$t_0$", "t0"))
    kepler6.parameters.append(pars.CosParameter(r"$\cos\,i$", "iobs"))

    # Add the datasets.
    kepler6.add_dataset(lc)
    # kepler6.add_dataset(sc)

    # Perturb the initial guess a bit.
    kepler6.planets[0].a *= 1 + 1e-3 * np.random.randn()
    kepler6.planets[0].r *= 1 + 1e-2 * np.random.randn()
    kepler6.planets[0].t0 = 1e-3 * np.random.randn()
    kepler6.iobs += np.random.rand() - 0.5

    # Run MCMC.
    kepler6.run_mcmc(2000, thin=10)

    # Plot the results.
    results = ResultsProcess(burnin=30)
    # results = kepler6.results(burnin=30)

    mean_a = results.semimajors[0] / results.rstar
    results.corner_plot([
        Column(r"$(a/R_\star - {0:.3f})\times10^{{3}}$".format(mean_a),
               lambda s: 1e3 * (s.planets[0].a / s.star.radius - mean_a)),
        Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
        Column(r"$t_0$", lambda s: s.planets[0].t0),
        Column(r"$i$", lambda s: s.iobs),
    ])
    results.time_plot()
