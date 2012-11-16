**THIS IS RESEARCH SOFTWARE THAT DOESN'T REALLY WORK YET. USE AT YOUR OWN
RISK**

*BART* is a set of code for modeling the light curves of exoplanet transits.

The core light curve routines are written in Fortran and wrapped in Python.


Installation
------------

Just clone and install:

::

    git clone https://github.com/dfm/bart.git
    cd bart
    python setup.py install

You'll need ``numpy`` and a Fortran compiler.


Usage
-----

To generate fake data for a "Hot Jupiter",

::

    import numpy as np
    import matplotlib.pyplot as pl
    import bart

    # The radius in Solar radii and the un-occulted flux of the star.
    rs, fs = 1.5, 100.0

    # The oservation angle of the planetary disk in degrees (zero is edge-on
    # just to be difficult).
    iobs = 0.01

    # The limb-darkening parameters.
    ld_type = "quad"
    nbins, gamma1, gamma2 = 50, 0.39, 0.1

    # Initialize the planetary system.
    system = bart.BART(rs, fs, iobs, ldp=[nbins, gamma1, gamma2], ldptype=ld_type)

    # The parameters of the planet:
    r = 5.0      # The radius of the planet in Jupter radii.
    a = 0.05     # The semi-major axis of the orbit in AU.
    e = 0.01     # The eccentricity of the orbit.
    T = 4.2      # The period of the orbit in days.
    phi = np.pi  # The phase of the orbit in radians.
    i = 0.0      # The relative observation angle for this planet in degrees.

    # Add the planet.
    system.add_planet(r, a, e, T, phi, i)

    # Compute a lightcurve.
    time = 365.0 * np.random.rand(500)
    ferr = 5 * np.random.rand(len(time))  # The uncertainties.
    flux = system.lightcurve(time) + ferr * np.random.randn(len(time))

    # Plot it.
    pl.errorbar(time, flux, yerr=ferr, fmt=".k")
    pl.xlabel("Time [days]")
    pl.ylabel("Flux")
    pl.savefig("lc.png")
