Flexible and rapid exoplanet transit modeling in Python with Bart
=================================================================

With enormous numbers of transiting exoplanets being discovered by
Kepler, it is finally becoming possible to hierarchically infer the
parameters of the global population of exoplanets.
A necessary component of these analyses is a tool for high performance
probabilistic transit modeling—sampling the posterior PDF for the
parameters of an arbitrary multi-planet exoplanet system given stellar
light curve data and parameterized priors.
We present a computationally efficient and extremely flexible system
called Bart for performing these inferences and marginalizations.
Bart uses a non-parametric limb darkening law for each star; it is much
more flexible than the classic analytic results in current use, at
minimal computational cost.
It is also possible, using Bart, to simultaneously model light curves
with varying photometric properties—for example short and long cadence
Kepler observations—and radial velocity profiles.
Bart is an open-source Python module available under the MIT license at
http://dan.iel.fm/bart.
