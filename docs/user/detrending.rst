.. _detrending:
.. module:: bart
.. highlight:: python

De-trending Kepler Light Curves
===============================

The raw aperture photometry *huge* (compared to the photometric precision)
long timescale variability. This is due to some combination of intrinsic
stellar variability and systematic instrumental effects. The Kepler team
releases the raw light curves and measurements that have been passed through
the Presearch Data Conditioning (PDC) pipeline. The PDC pipeline attempts to
remove some of the large scale variability but there remain significant
artifacts that are possible to remove. **Bart** includes a tool for robustly
removing many of these effects in a preprocessing step. It would be far more
rigorous to forward model the trends—perhaps using a Gaussian process—and we
have spent some time working on this problem but it isn't really
computationally tractable at this point so these techniques won't be discussed
here.


How to Use Bart to Apply De-trending
------------------------------------

By default the :class:`bart.dataset.KeplerDataset` applies spline de-trending
algorithm to the light curve when it reads the data. This behavior can be
specified using the ``detrend`` keyword argument in the constructor.

If you would like the apply the de-trending algorithm to a different dataset
or if you would like to have a greater level of control, you can use the
:func:`bart.kepler.spline_detrend` function.


Nature of the Trends
--------------------

There are three distinct classes of artifacts in the Kepler light curves:

1. smooth trends with timescales of a few days,
2. temporal breaks without data, and
3. abrupt discontinuities in sensitivity.

The following figure shows the long cadence data from a single quarter spent
observing the Kepler-6 system. The top panel shows the raw aperture photometry
reported by the Kepler pipeline. The middle panel shows the resulting light
curve after the PDC correction. The bottom panel shows the result of
de-trending using the techniques described here.

.. image:: ../_static/detrend.png

This is a nice example because it clearly displays all three issues mentioned
above. The smooth component seems roughly piecewise linear with the breaks in
the slope occurring where there are discontinuities in the data. At around 182
days, there is a temporal break (type 2 above) that also has a corresponding
sensitivity discontinuity (type 3 above). Then, at around 200 days, there is a
discontinuity that doesn't have a corresponding break in the time series. As
you can see from the middle panel, the PDC pipeline does not remove all the
artifacts—and no-one claims that it is supposed to—but we are able to catch
all of the problems using our slightly more sophisticated technique.


Fitting A Cubic Spline Using Iteratively Re-Weighted Least Squares
------------------------------------------------------------------

Our base model for de-trending is a cubic spline with knots approximately
every four days. This is clearly an over-constrained linear system so we will
use least squares (LS) to find the minimum :math:`\chi^2` solution. In an
example like the one in the figure above, naïve use of least-squares would
result in the fit being biased low because of the transits. To deal with this,
we use a robust algorithm called iteratively re-weighted least squares (IRLS;
CITE HOGG'S BRAIN?). This techniques is similar to sigma clipping but it is
somewhat more robust because it smoothly reduces the impact of outliers
without discounting them entirely (COME ON). The basic algorithm is as
follows:

1. Compute the inverse variance of the observations :math:`w_n=1/\sigma_n^2`
   where :math:`\sigma_n` are the quoted uncertainties.
2. Compute the LS solution weighted by :math:`w_n`.
3. Compute :math:`\chi_n^2 = [y_n - \hat{y}(x_n)]^2 / \sigma_n^2`.
4. Update the weights :math:`w_n = Q / [\sigma_n^2 \, (\chi_n^2 + Q)]` for
   some value of :math:`Q`.
5. Go to step 2 and iterate to convergence.

The choice of :math:`Q` sets the maximum effect that an outlier can have on
the fit. For an extreme outlier (:math:`\chi_n^2 \gg Q`), the effect of that
point will be down-weighted by the factor :math:`Q / \chi_n^2`. For points
that are well fit by the model, the weight is simply :math:`1/\sigma_n^2` as
you would use in standard LS. We find that a :math:`Q` of 4 seems to work well
in this context but there is no reason to assume that this will be an optimal
choice in all situations.

Now, let's return to the Kepler-6 example shown in the figure above. The
following figure shows the result of naïvely fitting the light curve using
a cubic spline and IRLS.

.. image:: ../_static/detrend_2.png

The top panel shows the raw aperture photometry as black points and the spline
fit is shown with a red line and the locations of the knots placed every four
days are shown as red dots. The bottom panel is the same data with the model
fit divided out. It's clear that we've been able to remove the global trends
but the behavior at the discontinuities is definitely still a significant
problem that we will deal with in the next section.


Dealing With Discontinuities
----------------------------

The results shown in the previous figure made it clear that the effects of
data artifacts of the types 2 and 3 are significant and a satisfactory
de-trending algorithm should handle them well. The temporal discontinuities
(type 2) are the easiest to deal with. Since the datasets have a well-defined
cadence, it is easy to pick out the discontinuities by examining the time
between subsequent observations in the dataset. If this time is substantially
longer than the cadence then this is a discontinuity that should be dealt
with. We found that a reasonable tolerance threshold is triple the cadence and
this seems to pick out most of the artifacts with very little contamination
but this choice was definitely far from systematic.

After finding all of the times where there is a discontinuity, we add a knot
at either end of the break and two evenly spaced in between.

.. image:: ../_static/detrend_3.png

The panels show the same information as the previous example and if you
compare them closely, you'll see that the problems caused by the breaks at
182 days and 230 days have been eliminated. There are, however, still a few
other problematic points. In particular, lets zoom in on the residuals away
from the fit at around 245 days. In the following figure, the top panel shows
the value of :math:`\chi_n = [y_n - \hat{y}(x_n)] / \sigma_n` at each
data point.
The discontinuity in the data is very visible in the :math:`\chi_n` plot and
it seems reasonable to approximately fit the shape of the :math:`\chi_n`
function near the discontinuity with the function:

.. math::
   k(t;\,t_0) = \left \{ \begin{array}{ll}
      [(t - t_0)/\delta - 1]^2 & 0 \le t-t_0 \le \delta \\
      -[(t - t_0)/\delta + 1]^2 & -\delta \le t-t_0 < 0 \\
      0 & \mathrm{otherwise}\\
   \end{array} \right .

for :math:`t_0 \approx 245\,\mathrm{days}` and
:math:`\delta \approx 2\,\mathrm{days}`. For comparison, the middle panel in
the following figure shows :math:`k(t;\,t_0)` plotted for :math:`t_0 = 246.2`
(which just happens to be the best value).

Then, we can define the scalar

.. math::
   S(t_0) = \frac{\vec{k}^\mathrm{T} \cdot \vec{r}}
                 {\vec{k}^\mathrm{T} \cdot \vec{k}}

where

.. math::
   \vec{k} = \left ( \begin{array}{ccc}
       k(t_1;\,t_0) & \cdots & k(t_N;\,t_0)
   \end{array} \right)^\mathrm{T}

and

.. math::
   \vec{r} = \left ( \begin{array}{ccc}
        \sqrt{\frac{Q}{Q + \chi_1^2}}\,\chi_1 & \cdots &
        \sqrt{\frac{Q}{Q + \chi_N^2}}\,\chi_N
   \end{array} \right)^\mathrm{T}\quad.

As with the IRLS discussion, :math:`\vec{r}` is the robustly re-weighted
error vector. With this definition, :math:`S^2` robustly increases for errors
in the data that have roughly the shape given by the kernel function
:math:`k(t;\,t_0)`. The bottom panel in the following figure shows the value
of :math:`S^2` computed at the midpoint between each pair of neighboring
data points.

.. image:: ../_static/detrend_5.png

Problems with the data that are not accounted for within the
re-weighted error bars will have values :math:`S^2 > Q`. Therefore, we can
automatically detect discontinuities like this one by iteratively finding the
time of the point with largest :math:`S^2 > Q`. At this breakpoint, we add a
knot to the spline at the position of the two neighboring data points and two
others evenly spaced in between. Then, we repeat the IRLS fit until there are
no other discontinuities with :math:`S^2 > Q`. The result in this example is
shown in the bottom panel of the following figure:

.. image:: ../_static/detrend_4.png

In Bart, this procedure is implemented in Fortran with Python bindings —
:func:`bart.kepler.spline_detrend` — and it is efficient enough to be run on
any large Kepler light curve.
