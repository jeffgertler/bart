import os
import sys

import numpy as np
import matplotlib.pyplot as pl
import pyfits

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, "..", ".."))
import bart


if __name__ == "__main__":
    f = pyfits.open(os.path.join(dirname, "data.fits"))
    lc = np.array(f[1].data)
    f.close()

    t = lc["TIME"]
    f, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    model = bart.fit_lightcurve(t, f, ferr, rs=1.487, p=0.357, T=3.2135,
                                            a=0.04558)
    t, f, ivar = model._data

    fit = model.lightcurve()
    T = model.tp[0]

    pl.plot(t % T, f, ".k")
    pl.plot(t % T, fit, "+r")
    pl.savefig("kepler4b.png")
