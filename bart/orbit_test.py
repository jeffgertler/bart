import numpy as np
import _bart
import matplotlib.pyplot as pl

e, a, T, phi, pomega, i = 0.9, 14.0, 1.1, 0.5, np.pi, 0.0
t = np.linspace(0, T, 100)
pos = _bart.solve_orbit(t, e, a, T, phi, pomega, i)

pl.axis("equal")
pl.plot(pos[0], pos[1], ".k")
pl.savefig("orbit.png")
