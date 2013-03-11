.. _model_building:
.. module:: bart
.. highlight:: python

Basic Model Building
====================

In this section, we'll discuss the model building syntax used by **Bart**. The
basic idea is that you start by declaring a full planetary system with the
number of planets and all the physical and observational parameters specified.
Then—starting from that point in parameter space—Bart can optimize or sample
the posterior probability distribution of any set of parameters given a set of
:class:`Dataset` objects. It is instructive to start by building a system and
generating fake data from the initial condition to see what it looks like. For
this purpose, we'll make a model that is roughly based on the planet
`Kepler-6b <http://kepler.nasa.gov/Mission/discoveries/kepler6b/>`_ discovered
by the *Kepler* satellite. To start we'll lay out the steps that are required
to generate a synthetic light curve and then—in the next section—we'll explain
how you would go about fitting for the parameters given this data.

*You can find the full source code for this example in the* `Bart source code
repository
<https://github.com/dfm/bart/blob/master/examples/docs/>`_.


The Star
--------

Kepler-6 is a metal-rich Sun-like star. The `Kepler-6b discovery paper
<http://arxiv.org/abs/1001.0333>`_ derives the following stellar parameters:

1. stellar mass :math:`M_\star = 1.209\,M_\odot`,
2. stellar radius :math:`R_\star = 1.391\,R_\odot`,
3. surface gravity :math:`\log_{10} g_\star (cgs) = 4.236`,
4. effective temperature :math:`T_\star = 5647\,K`, and
5. metallicity :math:`[\mathrm{Fe/H}] = +0.34`.

The metallicity, temperature and surface gravity suggest a particular shape of
the limb darkening profile in the Kepler band based on simulations of stellar
evolution and stellar atmospheres. To get this model using Bart, you run:

::

    from bart import kepler
    ldp = kepler.fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)

This returns a :class:`LimbDarkening` object that follows the quadratic form
from `Sing (2009) <http://arxiv.org/abs/0912.2274>`_. You can also use the
values from `Claret & Bloemen (2011)
<http://adsabs.harvard.edu/abs/2011A%26A...529A..75C>`_:

::

    ldp = fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50, model="claret11")

*Note*: This will take a long time the first time you run this command because
it will first need to download the data tables from the paper. This also means
that the first time that you run this command, you'll need to be connected to
the internet. This is not true for the Sing (2009) model because it is
included with the default build of Bart.

Now that we've determined the limb darkening profile, we can then initialize
the :class:`Star`. To do this, run:

::

    import bart
    star = bart.Star(mass=1.209, radius=1.391, ldp=ldp)


The Planet
----------

The `same discovery paper <http://arxiv.org/abs/1001.0333>`_ also derives the
orbital and physical parameters of the planet Kepler-6b based on the light
curve and radial velocity measurements. These parameters are:

1. period :math:`P = 3.234723\,\mathrm{days}`,
2. scaled semi-major axis :math:`a/R_\star = 7.05`,
3. scaled planet radius :math:`R_p/R_\star = 0.09829`,

To initialize the :class:`Planet`, you can just add:

::

    a = 7.05 * star.radius
    Rp = 0.09829 * star.radius
    planet = bart.Planet(a=a, r=Rp)

At this point, let's check to make sure that the parameters are all consistent
with the measured period:

::

    print(planet.get_period(star.mass))

This should print something like ``3.23343650114`` which is close enough for
our purposes. It is sometimes useful, however, to initialize the planet first
and then set the star mass using the :func:`get_mstar` method on the
:class:`Planet` object to ensure that the data will have the right period.


Putting it All Together
-----------------------

The :class:`Star` and :class:`Planet` are brought together by adding them to a
:class:`PlanetarySystem`. You can also specify the inclination of the orbital
plane when creating the system. For Kepler-6b, the inclination was found to be
:math:`86.8^\circ`. Therefore, you can build the system as follows:

::

    kepler6 = bart.PlanetarySystem(star, iobs=86.8)
    kepler6.add_planet(planet)

and then plot the model light curve:

::

    import numpy as np
    import matplotlib.pyplot as pl

    t = np.linspace(-0.2, 0.2, 5000)
    pl.plot(t, kepler6.lightcurve(t))

This should result in a plot that looks something like this:

.. image:: ../_static/model_building.png


Generating Synthetic Data
-------------------------

Now, we'll generate some fake data that mimics long and short cadence light
curves observed by Kepler. Short cadence data are integrated for 54.2 seconds.
Including readout, the entire exposure lasts 58.9 seconds. The long cadence
data have 1626 second integrations in 1766 second exposures. To generate some
synthetic long cadence data for our Kepler-6 model, we can generate some
samples from the model at the correct cadence:

::

    lc_time = np.arange(0, 90., 1766 / (60. * 60. * 24.))
    lc_flux = kepler6.lightcurve(lc_time, texp=1626)

and then add some representative observational uncertainties:

::

    lc_err = 1.5e-3 * np.random.rand(len(lc_flux))
    lc_flux = lc_flux + lc_err * np.random.randn(len(lc_flux))

The same procedure for the short cadence data looks like:

::

    sc_time = np.arange(0, 90., 58.9 / (60. * 60. * 24.))
    sc_flux = kepler6.lightcurve(sc_time, texp=54.2)
    sc_err = 3e-3 * np.random.rand(len(sc_flux))
    sc_flux = sc_flux + sc_err * np.random.randn(len(sc_flux))

A plot of these synthetic datasets should look something like:

.. image:: ../_static/model_building_data.png


Fitting Light Curve Data
------------------------


