.. _detrending:
.. module:: bart
.. highlight:: python

De-trending Kepler Light Curves
===============================

The raw aperture photometry *huge* (compared to the photometric precision)
long timescale variability. This is due to some combination of intrinsic
stellar variability and systematic instrumental effects. The Kepler team
releases the raw light curves and measurements that have been passed through
the Presearch Data Conditioning (PDC) pipeline. The PDC pipeline attempts to
remove some of the large scale variability but there remain significant
artifacts that are possible to remove. **Bart** includes a tool for robustly
removing many of these effects in a preprocessing step. It would be far more
rigorous to forward model the trends—perhaps using a Gaussian process—and we
have spent some time working on this problem but it isn't really
computationally tractable at this point so these techniques won't be discussed
here.


Nature of the Trends
--------------------

There are three distinct classes of artifacts in the Kepler light curves:

1. smooth trends with timescales of a few days,
2. temporal breaks without data, and
3. abrupt discontinuities in sensitivity.

The following figure shows examples
