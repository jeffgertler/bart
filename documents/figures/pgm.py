#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

# Import the LaTeX variables.
rc("text.latex", preamble=["\input{"
    + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "vars.tex") + "}"])

import daft

pgm = daft.PGM([4, 5.5], origin=[0.75, 2.25])

pgm.add_node(daft.Node("planet_pars",
                       r"\tzero, \smaxis, \ecc, \pomega, \incl, \mplanet",
                       3.5, 7, aspect=4.0))
pgm.add_node(daft.Node("xt", r"$\bvec{x}(t)$", 3, 6, aspect=1.1))
pgm.add_edge("planet_pars", "xt")

pgm.add_node(daft.Node("mstar", r"\mstar", 1.5, 7))
pgm.add_edge("mstar", "xt")

pgm.add_node(daft.Node("rp", r"\rplanet", 4, 6.25))
pgm.add_node(daft.Node("rstar", r"\rstar, \fstar, \setof{I_k}", 1.5, 6,
                       aspect=2.5))
pgm.add_node(daft.Node("f", r"\fmodel", 3, 4, aspect=1.4))
pgm.add_edge("rp", "f")
pgm.add_edge("rstar", "f")
pgm.add_edge("xt", "f")

pgm.add_node(daft.Node("tn", r"\tobs", 4, 5, observed=True))
pgm.add_node(daft.Node("texp", r"${t_\mathrm{exp}}_n$", 4, 4, observed=True))
pgm.add_edge("tn", "f")
pgm.add_edge("texp", "f")

pgm.add_node(daft.Node("iobs", r"\iobs", 1.5, 5))
pgm.add_edge("iobs", "f")

pgm.add_node(daft.Node("fobs", r"\fobs", 3, 3, observed=True,
                       aspect=1.4))
pgm.add_node(daft.Node("ferr", r"\ferr", 4, 3, observed=True))
pgm.add_node(daft.Node("jitter", r"\jitter", 1.5, 4))
pgm.add_edge("f", "fobs")
pgm.add_edge("ferr", "fobs")
pgm.add_edge("jitter", "fobs")

pgm.add_plate(daft.Plate([2.4, 2.4, 2.2, 3], label="observations",
                         position="bottom right"))
pgm.add_plate(daft.Plate([2.4, 5.5, 2.2, 2], label="planets",
                         position="bottom right"))

pgm.render()
pgm.figure.savefig("pgm.pdf")
