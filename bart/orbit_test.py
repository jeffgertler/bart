import numpy as np
import _bart
import matplotlib.pyplot as pl

# import astropy.units as u
# from astropy.constants import si
# G = si.G.to(u.R_sun ** 3 / (u.M_sun * u.day ** 2)).value
G = 2945.4625385377644

mstar = 1.0
e, a, i = 0.6, 14.0, 0
t0 = 0.15
pomega = 1.5 * np.pi

T = 2 * np.pi * np.sqrt(a * a * a / G / mstar)
t = np.linspace(0, T, 10000)

pos = _bart.solve_orbit(t, mstar, e, a, t0, pomega, i, 0.0)
pphi = _bart.solve_orbit(t0, mstar, e, a, t0, pomega, i, 0.0)
ee = np.arccos((e + np.cos(pomega)) / (1 + e * np.cos(pomega)))

ax = pl.figure().add_axes((0, 0, 1, 1), frameon=False,
                        xticks=[], yticks=[], aspect="equal")

ax.plot(pos[0], pos[1], "k")

# Plot circular orbit.
f = e * a
center = f * np.array([-np.cos(pomega), np.sin(pomega)])
ax.plot(center[0], center[1], ".k")

# Plot axis of ellipse.
xax = np.array([[-(a + f) * np.cos(pomega), (a + f) * np.sin(pomega)],
                [(a - f) * np.cos(pomega), -(a - f) * np.sin(pomega)]])
ax.plot(xax[:, 0], xax[:, 1], "k")

# Plot phi axis.
# phiax = 5 * np.array([[0, 0],
#                   [np.cos(0.25 * np.pi + phi), -np.sin(0.25 * np.pi + phi)]])
# ax.plot(phiax[:, 0], phiax[:, 1], "k")

# Plot key points.
ax.plot(pos[0, 0], pos[1, 0], "or")
ax.plot(pphi[0], pphi[1], "og")
# ax.plot(ppom[0], ppom[1], "ob")

ax.axhline(0, color="k")
ax.axvline(0, color="k")

ax.set_ylim(1.1 * np.array(ax.get_ylim()))
ax.set_xlim(np.array([1.1, 1.2]) * np.array(ax.get_xlim()))

pl.savefig("orbit.png")
