Rapid Exoplanet Transit Modeling in Python
==========================================

**Bart** is a set of tools for working with observations of `exoplanets
<http://en.wikipedia.org/wiki/Extrasolar_planet>`_. There are two main
components: a standalone **Fortran library** for quickly generating extremely
general models of exoplanet transits and radial velocity observations, and
**Python bindings** to this library that make it extremely easy to fit for
the physical parameters of an exoplanetary system given a heterogeneous set
of observations.


Features
--------

- Built-in integration with the Kepler data repository
- General non-parametric limb-darkening
- Sophisticated but user-friendly model building syntax
- Efficient MCMC sampling using `emcee <http://dan.iel.fm/emcee>`_

User Guide
----------

.. toctree::
   :maxdepth: 2

   user/quickstart
   user/kepler
   user/faq
   js

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api


Authors & Contributions
-----------------------

**Bart** is being developed and supported by `Dan Foreman-Mackey
<http://dan.iel.fm>`_.

For the hackers in the house, development happens on `Github
<https://github.com/dfm/bart>`_ and we welcome pull requests.


License
-------

*Copyright 2013 Dan Foreman-Mackey & contributors.*

**Bart** is free software made available under the *MIT License*. For details
see `the LICENSE file <https://raw.github.com/dfm/bart/master/LICENSE.rst>`_.
