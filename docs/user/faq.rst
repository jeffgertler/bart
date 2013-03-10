.. _faq:
.. module:: bart
.. highlight:: bash

FAQ
===

How to set up a virtual environment
-----------------------------------

It's generally a good idea to use virtual environments to manage your Python
dependencies. `virtualenv <http://www.virtualenv.org/>`_ is a command line
script that helps you set up your environment. After you have `installed
virtualenv <http://www.virtualenv.org#installation>`_, navigate into the
directory where you plan on using Bart. Then run:

::

    virtualenv venv --distribute
    source venv/bin/activate

To set up and activate your new virtual environment. If you're using Mac OS X,
it's good to set the environment variable ``CC=clang`` now. Then you should
install `numpy <http://www.numpy.org/>`_:

::

    pip install numpy>=1.7.0

If that works, `install Bart <../quickstart>`_.
