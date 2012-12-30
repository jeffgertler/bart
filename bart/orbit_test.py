import numpy as np
import _bart
import matplotlib.pyplot as pl

e, a, T, i = 0.65, 14.0, 1.1, 0.0
phi = 0.07 * np.pi
pomega = 0.1 * np.pi
t = np.linspace(0, T, 10000)

pos = _bart.solve_orbit(t, e, a, T, phi, pomega, i)
pphi = _bart.solve_orbit(phi * T / 2 / np.pi, e, a, T, phi, pomega, i)

ee = np.arccos((e + np.cos(pomega)) / (1 + e * np.cos(pomega)))
t0 = (ee - e * np.sin(ee) + phi) * T / 2 / np.pi

ppom = _bart.solve_orbit(t0, e, a, T, phi, pomega, i)

pl.axis("equal")
ax = pl.gca()

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
phiax = a * (1 - e * e) / (1 + e * np.cos(pomega + phi)) * np.array([[0, 0],
                [np.sin(0.5 * np.pi - pomega - phi),
                 -np.cos(0.5 * np.pi - pomega - phi)]])
ax.plot(phiax[:, 0], phiax[:, 1], "k")

# Plot key points.
ax.plot(pos[0, 0], pos[1, 0], "or")
ax.plot(pphi[0], pphi[1], "og")
ax.plot(ppom[0], ppom[1], "ob")

ax.axhline(0, color="k")
ax.axvline(0, color="k")

pl.savefig("orbit.png")
