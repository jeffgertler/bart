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

Next, set up your `virtual environment <http://www.virtualenv.org/>`_:

::

    virtualenv venv --distribute
    source venv/bin/activate

Then, install the prerequisites (**note**: *on Mac OS, it's worth setting the
environment variable* ``CC=clang``):

::

    pip install 'numpy>=1.6.2'
    pip install -r requirements.txt

Finally, either build the Fortran and C extensions:

::

    python setup.py interface
    python setup.py build_ext --inplace


Usage
-----

Take a look at `this example <https://github.com/dfm/bart/blob/master/examples/quick/quick.py>`_.
