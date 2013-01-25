#!/usr/bin/env python
"""
Gaussian mixture models

"""

from __future__ import division, print_function


__all__ = ['MixtureModel']


import numpy as np

from . import _algorithms


class MixtureModel(object):
    """
    Gaussian mixture model for samples.

    ``P`` data points in ``D`` dimensions with ``K`` clusters.

    :param K:
        The number of Gaussians to include in the mixture.

    :param data:
        A ``P x D`` ``numpy.ndarray`` of the samples.

    """
    def __init__(self, K, data, init_grid=False):
        self.K = K
        self._data = np.atleast_2d(data)
        self._lu = None

        self.kmeans_rs = np.zeros(self._data.shape[0], dtype=int)

        # Randomly choose ``K`` components to be the initial means.
        inds = np.random.randint(data.shape[0], size=self.K)
        self.means = data[inds, :]

        # Initialize the covariances as the data covariance.
        self.cov = np.array([np.cov(data, rowvar=0)] * self.K)

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

    def run_kmeans(self, maxiter=200, tol=1e-4, verbose=False):
        """
        Run the K-means algorithm using the C extension.

        :param maxiter:
            The maximum number of iterations to try.

        :param tol:
            The tolerance on the relative change in the loss function that
            controls convergence.

        :param verbose:
            Print all the messages?

        """
        iterations = _algorithms.kmeans(self._data, self.means,
                self.kmeans_rs, tol, maxiter)

        if verbose:
            if iterations < maxiter:
                print("K-means converged after {0} iterations."
                        .format(iterations))
            else:
                print("K-means *didn't* converge after {0} iterations."
                        .format(iterations))
