.. _quickstart:

.. module:: bart

Getting Started
===============

This page provides basic instructions for getting started with **Bart**.


Installation
------------

The easiest way to install Bart is with the standard Python build system
`pip <http://www.pip-installer.org/>`_ but you can also install from source.
First, you'll need to make sure that you have `numpy <http://www.numpy.org>`_
and a Fortran compiler installed on your system. We recommend using a `virtual
environment <../faq>`_.


Using pip
*********

If you have numpy and a Fortran compiler installed, you should be able to run

::

    pip install bart

to build and install the `most recent stable version of Bart
<http://pypi.python.org/pypi/bart>`_. To get the development version, you can
run

::

    pip install -e git+https://github.com/dfm/bart.git#egg=bart-dev

or build from the source.


Source installation
*******************

If you have `git <http://git-scm.com/>`_ installed, you can get the most
recent version of the source code by cloning `the repository
<https://github.com/dfm/bart>`_

::

    git clone https://github.com/dfm/bart.git
    cd bart

Otherwise, the source can be downloaded as `a .zip archive
<https://github.com/dfm/bart/archive/master.zip>`_

::

    curl -L https://github.com/dfm/bart/archive/master.zip -o bart.zip
    unzip bart.zip
    cd bart-master

Then, from within the source directory, run

::

    python setup.py install

to build and install Bart.


Testing the installation
************************

Nothing to see here yet.


A Simple Example
----------------

The following sections demonstrate how you can use Bart to generate some
mock exoplanet observations and then fit them using MCMC.
