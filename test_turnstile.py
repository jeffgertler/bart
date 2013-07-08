#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from bart.injection import kepler_injection
from bart._turnstile import period_search

datasets, ps = kepler_injection(2301306, 278.0, 0.03, t0=20.0)

print(period_search(datasets, 277., 279., 5, 0., 0.05 ** 2, 10, 1.0, 1.0))
