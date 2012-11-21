**THIS IS RESEARCH SOFTWARE THAT DOESN'T REALLY WORK YET. USE AT YOUR OWN
RISK**

.. image:: https://raw.github.com/dfm/bart/master/logo/logo.png

*BART* is a set of code for modeling the light curves of exoplanet transits.

The core light curve routines are written in Fortran and wrapped in Python.


Installation
------------

First, clone:

::

    git clone https://github.com/dfm/bart.git
    cd bart

Then, install the prerequisites:

::

    pip install numpy
    pip install scipy matplotlib
    pip install -r requirements.txt

Finally, install and profit:

::

    python setup.py install

You'll need ``numpy`` and a Fortran compiler.


Usage
-----

Take a look at `this example <https://github.com/dfm/bart/blob/master/examples/quick/quick.py>`_.
