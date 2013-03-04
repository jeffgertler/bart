.. _kepler:

.. module:: bart

Working With Kepler Data
========================

**Bart** was designed with *Kepler* in mind. As
a result, there are a lot of tools built into Bart that make interacting
with *Kepler* data a pleasure instead of a pain in the ass (we're trying, at
least).


Interacting with the MAST API
-----------------------------

`MAST <http://archive.stsci.edu/>`_ provides a basic API to access all the
cataloged results from the *Kepler* team. Bart includes a Python module that
interacts with this API providing a few convenient features. The main
interaction is by way of the :class:`kepler.API` object. The documentation for
the API itself can be found on the `MAST website
<http://archive.stsci.edu/vo/mast_services.html>`_. At this point, the built
in Python wrapper only officially supports the `KOI
<http://archive.stsci.edu/search_fields.php?mission=kepler_koi>`_ and
`confirmed planet
<http://archive.stsci.edu/search_fields.php?mission=kepler_cp>`_ tables but it
can still be used to interact with the other tables.

To use this interface, you start by initializing the API interface:

::

    from bart import kepler
    api = kepler.API()


Finding confirmed planets
*************************

To get a list of all of the confirmed planets, you can call the
:func:`planets` method:

::

    planets = api.planets()
    for p in planets:
        print(p["kepler_name"], p["koi"], p["kepid"])

That snippet should show a list of the names, KOI numbers, and Kepler KIC IDs
for all of the confirmed planets in the database. A complete list of all of
the available fields is given on the `MAST documentation page
<http://archive.stsci.edu/search_fields.php?mission=kepler_cp>`_. You can also
search the API (using any of the fields listed on the documentation page):

::

    planets = api.planets(kepid=9821454)
    for p in planets:
        print(p["kepler_name"], p["period"])

This snippet will list the names and periods (in days) of the two planets in
the Kepler-37 system. You can also run inequality searches. For example, you
can find all the planets with periods longer than 50 days by running:

::

    planets = api.planets(period=">50")


Finding planet candidates
*************************

The planet candidate list (`Batalha et al. (2013)
<http://arxiv.org/abs/1202.5852>`_) is provided in the `Kepler Objects of
Interest (KOI)
<http://archive.stsci.edu/search_fields.php?mission=kepler_koi>`_ table. This
table can be queried using the :func:`kois` method:

::

    kois = api.kois()
    for k in kois:
        print(koi["kepoi"], koi["period"])

This table can also be filtered on any of the fields listed in the
`MAST documentation
<http://archive.stsci.edu/search_fields.php?mission=kepler_koi>`_.


A note about generators
***********************

You might be surprised by the result if you try to access a specific KOI or
planet candidate returned by one of the above commands like you would normally
access a list element:

::

    planets = api.planets()
    print(planets[1])

This will throw a ``TypeError`` instead of printing out all the planet
parameters for the second planet returned by the query. This is because the
API calls all return `generators
<http://docs.python.org/2/tutorial/classes.html#generators>`_. This was
necessary because the MAST API doesn't really support pagination natively so
some hacking was needed. As a result you need to access the elements by
looping over them or by using the :func:`next` method:

::

    planets = api.planets()
    print(planets.next())


Downloading Kepler Light Curves
-------------------------------

Bart's Kepler API layer also supports the downloading of raw Kepler data. To
grab all the files for a particular star, you can run:

::

    from bart import kepler
    api = kepler.API()

    filenames = api.data(9821454).fetch_all("data")

This will find and download all of the light curves for the Kepler star
9821454 and save them as FITS files in the directory ``data``. The result is a
list of the filenames for these files. Any light curves that already exist in
the chosen location, they won't be overwritten but the filenames will still be
returned so you won't need to special case that.

Bart also includes an interface for reading these light curve files:

::

    from kepler.dataset import KeplerDataset
    ds = KeplerDataset(filenames[0])
