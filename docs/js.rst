Bart.js — Exoplanet Transits in Your Browser
============================================

.. raw:: html

    <link rel="stylesheet" href="../_static/transit.css" type="text/css">

    <div id="plot"></div>

    <script src="http://d3js.org/d3.v3.min.js"></script>
    <script src="../_static/bart.js"></script>
    <script src="../_static/transit.js"></script>

This is a visualization of how an exoplanet transit observation works. The
top-left panel shows a sketch of the (to scale) orbit of a "hot Jupiter"
(based roughly on `Kepler-6b <http://arxiv.org/abs/1001.0333>`_) around a
Sun-like star. The top-right panel shows the same orbit from the point of
view of an observer that is nearly aligned with the orbital plane. In
reality, we can't actually spatially resolve a system like this. Instead, we
can only measure the brightness of the star as a function of time—the "light
curve". The bottom panel shows the true light curve as a line and an example
of what measurements (from the Kepler satellite, for example) might look
like.

Here's the code...

.. literalinclude:: _static/bart.js
   :language: javascript
