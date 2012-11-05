# Text defaults.
from matplotlib import rc
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

import numpy as np

import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrow


class Context(object):
    """
    A rendering context to handle plot coordinates (based on code from daft).

    :param shape:
        The number of rows and columns in the grid.

    :param origin:
        The coordinates of the bottom left corner of the plot.

    :param unit:
        The size of the grid spacing measured in centimeters.

    """
    def __init__(self, shape=[1, 1], origin=[0, 0], unit=2.0):
        # Set up the figure and grid dimensions.
        self.shape = np.array(shape)
        self.origin = np.array(origin)
        self.unit = unit
        self.figsize = self.unit * self.shape / 2.54

        # Initialize the figure to ``None`` to handle caching later.
        self._figure = None
        self._ax = None

    @property
    def figure(self):
        if self._figure is not None:
            return self._figure
        self._figure = pl.figure(figsize=self.figsize)
        return self._figure

    @property
    def ax(self):
        if self._ax is not None:
            return self._ax

        # Add a new axis object if it doesn't exist.
        self._ax = self.figure.add_axes((0, 0, 1, 1), frameon=False,
                xticks=[], yticks=[])

        # Set the bounds.
        l0 = self.convert(*self.origin)
        l1 = self.convert(*(self.origin + self.shape))
        self._ax.set_xlim(l0[0], l1[0])
        self._ax.set_ylim(l0[1], l1[1])

        return self._ax

    def convert(self, *xy):
        assert len(xy) == 2
        return self.unit * (np.atleast_1d(xy) - self.origin)

    def add_ellipse(self, x, y, r, ry=None, **p):
        p["lw"] = _pop_multiple(p, 1.0, "lw", "linewidth")
        p["ec"] = p["edgecolor"] = _pop_multiple(p, "k", "ec", "edgecolor")
        p["fc"] = p["facecolor"] = _pop_multiple(p, "none", "fc", "facecolor")

        if ry is None:
            ry = r

        el = Ellipse(xy=self.convert(x, y),
                     width=2 * r * self.unit, height=2 * ry * self.unit,
                     **p)

        self.ax.add_artist(el)

    def add_line(self, x, y, a1=False, a2=False, text=None, **p):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        assert len(x) == len(y)

        # Line styles.
        p["lw"] = _pop_multiple(p, 1.0, "lw", "linewidth")
        p["color"] = p.pop("color", "k")

        # Arrow styles.
        a = {}
        a["ec"] = a["edgecolor"] = \
                                _pop_multiple(p, None, "ec", "edgecolor")
        a["fc"] = a["facecolor"] = \
                            _pop_multiple(p, p["color"], "fc", "facecolor")
        a["head_length"] = p.pop("head_length", 0.2)
        a["head_width"] = p.pop("head_width", 0.15)
        if a["ec"] is None:
            a["ec"] = a["edgecolor"] = a["fc"]

        # Text styles.
        va = p.pop("va", "baseline")
        offset = p.pop("offset", [0, 0])

        pts = np.array([self.convert(x[i], y[i]) for i in range(len(x))])

        self.ax.plot(pts[:, 0], pts[:, 1], **p)

        # Draw the arrows.
        if a1:
            [[x0, y0], [x1, y1]] = pts[:2]
            el = FancyArrow(x1, y1, x0 - x1, y0 - y1, width=0,
                            length_includes_head=True, **a)
            self.ax.add_artist(el)

        if a2:
            [[x1, y1], [x0, y0]] = pts[-2:]
            el = FancyArrow(x1, y1, x0 - x1, y0 - y1, width=0,
                            length_includes_head=True, **a)
            self.ax.add_artist(el)

        # Annotate.
        if text is not None:
            xy = np.mean(pts, axis=0)
            self.ax.annotate(text, xy, xycoords="data",
                             ha="center", va=va,
                             xytext=offset, textcoords="offset points")


def _pop_multiple(d, default, *args):
    assert len(args) > 0, "You must provide at least one argument to 'pop'."

    results = []
    for k in args:
        try:
            results.append((k, d.pop(k)))
        except KeyError:
            pass

    if len(results) > 1:
        raise TypeError("The arguments ({0}) are equivalent, you can only "
                .format(", ".join([k for k, v in results]))
                + "provide one of them.")

    if len(results) == 0:
        return default

    return results[0][1]
