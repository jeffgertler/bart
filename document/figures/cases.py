#!/usr/bin/env python


import numpy as np

from utils import Context


def main(case):
    ctx = Context(shape=[1.5, 1.2], origin=[0.9, 0.7], unit=3)

    # Draw the star slice.
    star = [1.0, 1.0]
    r1, r2 = 0.75, 0.95

    th = np.linspace(0, 0.25 * np.pi, 100)
    x, y = star[0] + r1 * np.cos(th), star[1] + r1 * np.sin(th)
    ctx.plot(x, y, "k", lw=2, solid_capstyle="butt")

    x, y = star[0] + r2 * np.cos(th), star[1] + r2 * np.sin(th)
    ctx.plot(x, y, "k", lw=2, solid_capstyle="butt")

    # Draw the planet.
    p = 0.4
    z = 1.2
    ctx.add_arc(star[0] + z, star[1], p, theta1=90.0, theta2=180.0,
                ls="dotted")

    # Fill the occulted region.
    thmax = np.arccos((z * z + p * p - r2 * r2) / (2 * p * z))
    th = np.linspace(0, thmax, 100)
    x, y = star[0] + z - p * np.cos(th), star[1] + p * np.sin(th)

    thmax2 = np.arccos((z * z - p * p + r2 * r2) / (2 * r2 * z))
    th2 = np.linspace(thmax2, 0.0, 100)
    x = np.append(x, star[0] + r2 * np.cos(th2))
    y = np.append(y, star[1] + r2 * np.sin(th2))

    ctx.fill(x, y, color="#dddddd")
    ctx.plot(x, y, "k")

    # Impact parameter.
    ctx.add_line([star[0], star[0] + z], [star[1], star[1]],
                a1=True, a2=True,
                text=r"$b$", va="top", offset=[0, -3],
                )

    # Planet radius.
    th = np.radians(90.)
    x, y = star[0] + z + p * np.cos(th), star[1] + p * np.sin(th)
    ctx.add_line([star[0] + z, x], [star[1], y],
                a1=False, a2=True,
                text=r"$a$", va="center", offset=[8, 0],
                )

    # Slice radii.
    th = np.radians(45.)
    x, y = star[0] + r1 * np.cos(th), star[1] + r1 * np.sin(th)
    x1, y1 = star[0] + r2 * np.cos(th), star[1] + r2 * np.sin(th)

    ctx.add_line([star[0], x], [star[1], y],
                a1=False, a2=True,
                text=r"$r_0$", offset=[-5, 3],
                padding=0.02,
                )
    ctx.add_line([x, x1], [y, y1],
                a1=True, a2=True,
                text=r"$\Delta$", offset=[-6, 5],
                padding=0.02,
                )

    ctx.figure.savefig("case{0}.pdf".format(case))


if __name__ == "__main__":
    import sys
    try:
        case = int(sys.argv[1])
    except IndexError:
        case = 1
    main(case)
