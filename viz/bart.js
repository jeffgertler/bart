(function () {

  "use strict";

  var bart = window.Bart = {};

  bart.LimbDarkening = function (bins, intensity) {
    // A limb darkening profile. `bins` is a list of outer bin edges and
    // `intensity` is a list of the same length with the intensity levels.
    // `bins` should be *sorted*.
    this.bins = bins.map(function (v) { return v / bins[bins.length - 1]; });

    // Normalize the intensity profile.
    var _this = this;
    var norm = Math.PI * intensity.reduce(function (n, v, i, a) {
      var r2 = _this.bins[i] * _this.bins[i];
      if (i === 0) return v * r2;
      return n + v * (r2 - _this.bins[i - 1] * _this.bins[i - 1]);
    });
    this.intensity = intensity.map(function (v) { return v / norm });
  };

  bart.occulted_area = function (R, p, b) {
    // Get the occulted area for a star of radius `R`, planet of radius `p`
    // and impact parameter `b`.
    if (b >= R + p) return 0.0;
    else if (b <= R - p) return Math.PI * p * p;
    else if (b <= p - R) return Math.PI * R * R;
    else {
      var R2 = R * R,
          p2 = p * p,
          b2 = b * b,
          k1 = Math.acos(0.5 * (b2 + p2 - R2) / b / p),
          k2 = Math.acos(0.5 * (b2 + R2 - p2) / b / R),
          k3 = Math.sqrt((p+R-b) * (b+p-R) * (b-p+R) * (b+R+p));
      return p2 * k1 + R2 * k2 - 0.5 * k3;
    }
  };

  bart.ldlc = function (p, b, ldp) {
    // Get the light curve for a planet of radius `p`, a list of impact
    // parameters `b` and a limb darkening profile.
    return b.map(function (b0) {
      var areas = ldp.bins.map(function (bin, j) {
        return bart.occulted_area(bin, p, b0);
      });
      return areas.reduce(function (result, a, j) {
        if (j === 0) return 1 - a * ldp.intensity[j];
        return result - ldp.intensity[j] * (a - areas[j - 1]);
      }, 1.0);
    });
  };

  bart.orbit = function (ts, period, a, t0, incl) {
    // Get the 3D position of a planet in a circular orbit at times `ts`. The
    // orbit has period `period`, semi-major axis `a`, initial transit time
    // `t0` and inclination `incl` (in radians).
    return ts.map(function (t) {
      var wt = 2 * Math.PI * (t - t0) / period,
          xp = a * Math.cos(wt), yp = a * Math.sin(wt);
      return [xp * Math.sin(incl), yp, xp * Math.cos(incl)];
    });
  };

  bart.Lightcurve = function (ts, period, a, t0, incl, p, ldp) {
    this.period = period;
    this.a = a;
    this.t0 = t0;
    this.incl = incl;
    this.p = p;
    this.ldp = ldp;
    this.pos = bart.orbit(ts, period, a, t0, Math.PI * incl / 180.);
    this.b = this.pos.map(function (p0) {
      if (p0[0] > 0) return Math.sqrt(p0[1]*p0[1]+p0[2]*p0[2]);
      return 1 + p;
    });
    this.time = ts;
    this.flux = bart.ldlc(p, this.b, ldp);
  };

  bart.fiducial_ldp = function (g1, g2) {
    // Get a quadratic limb darkening profile.
    var bins = [],
        intensity = [];
    for (var i = 1; i <= 25; ++i) {
      var r = i / 25.,
          onemmu = 1 - Math.sqrt(1 - r * r);
      bins.push(r);
      intensity.push(1 - g1 * onemmu - g2 * onemmu * onemmu);
    }
    return new bart.LimbDarkening(bins, intensity);
  };

})();
