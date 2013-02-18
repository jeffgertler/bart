#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["km1d"]


import numpy as np


def km1d(x, k=2, maxiter=256, miniter=2):
    means = x[np.random.randint(len(x), size=k)]
    r0 = None

    for j in range(maxiter):
        # Assignment step.
        rs = np.argmin(np.abs(x[None, :] - means[:, None]), axis=0)

        # Update step.
        means = np.array([np.mean(x[rs == i]) for i in range(k)])

        # Convergence.
        if j > miniter and np.all(rs == r0):
            break

        r0 = rs

    if j >= maxiter - 1:
        print("Warning: K-means didn't converge.")

    return means, rs


if __name__ == "__main__":
    x = np.append(1000 + np.random.randn(1000), 50 + np.random.randn(1))
    np.random.shuffle(x)
    print(km1d(x))
