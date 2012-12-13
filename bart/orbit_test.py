import numpy as np
import _bart
import matplotlib.pyplot as pl

e, a, T, phi, pomega, i = 0.9, 14.0, 1.1, 0.5, 0.9 * np.pi, 0.0
t = np.linspace(0, T, 100)

pos = _bart.solve_orbit(t, e, a, T, phi, pomega, i)
pphi = _bart.solve_orbit(phi * T / 2 / np.pi, e, a, T, phi, pomega, i)

ee = np.arccos((e + np.cos(pomega)) / (1 + e * np.cos(pomega)))
t0 = (ee - e * np.sin(ee) + phi) * T / 2 / np.pi

ppom = _bart.solve_orbit(t0, e, a, T, phi, pomega, i)

pl.axis("equal")

pl.plot(pos[0], pos[1], ".k")
pl.plot(pos[0, 0], pos[1, 0], "+r")
pl.plot(pphi[0], pphi[1], ".g")
pl.plot(ppom[0], ppom[1], ".b")

pl.gca().axhline(0)

pl.savefig("orbit.png")
