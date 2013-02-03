#!/usr/bin/env python


import numpy as np

from utils import Context


ctx = Context(shape=[2.8, 2.2], origin=[-0.1, -0.1], unit=2.5)

# Draw the inner annulus.
star = [1.0, 1.0]
radii = np.linspace(0., 1., 6)[1:] ** 0.5
for r in radii:
    ctx.add_ellipse(star[0], star[1], r, ec="#666666")
ctx.add_ellipse(star[0], star[1], 1)

# Draw the planet.
p = 0.4
z = 1.1
ctx.add_ellipse(star[0] + z, star[1], p, fc="#dddddd")

# Overlay the dashed star.
for r in radii[:-1]:
    ctx.add_ellipse(star[0], star[1], r, ls="dotted", ec="#666666")
ctx.add_ellipse(star[0], star[1], radii[-1], ls="dotted")

# Overlay the planet again.
ctx.add_ellipse(star[0] + z, star[1], p)

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
             text=r"$p$", va="top", offset=[3, -3],
            )

# Stellar radius.
th = np.radians(150.)
x, y = star[0] + np.cos(th), star[1] + np.sin(th)
ctx.add_line([star[0], x], [star[1], y],
             a1=True, a2=True,
             text=r"$R$", offset=[3, 3],
            )

ctx.figure.savefig("geom.pdf")
