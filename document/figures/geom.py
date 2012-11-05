#!/usr/bin/env python


import numpy as np

from utils import Context


ctx = Context(shape=[2.8, 2.2], origin=[-0.1, -0.1], unit=3)

# Draw the star.
star = [1.0, 1.0]
ctx.add_ellipse(star[0], star[1], 1)

# Draw the planet.
p = 0.4
z = 1.2
ctx.add_ellipse(star[0] + z, star[1], p, fc="#dddddd")

# Impact parameter.
ctx.add_line([star[0], star[0] + z], [star[1], star[1]],
             a1=True, a2=True,
             text=r"$b$", offset=[0, 3],
            )

# Planet radius.
th = np.radians(25.)
x, y = star[0] + z + p * np.cos(th), star[1] + p * np.sin(th)
ctx.add_line([star[0] + z, x], [star[1], y],
             a1=True, a2=True,
             text=r"$p$", offset=[-3, 3],
            )

# Stellar radius.
th = np.radians(150.)
x, y = star[0] + np.cos(th), star[1] + np.sin(th)
ctx.add_line([star[0], x], [star[1], y],
             a1=True, a2=True,
             text=r"$1$", offset=[3, 3],
            )

ctx.figure.savefig("geom.pdf")
