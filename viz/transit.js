var width = 5, height = 3,
    updating = false,
    current_bin = 3,
    a = 0.2, b = 0.5;

var scale = d3.scale.linear().domain([0, 1]).range([0, 100]),
    xscale = d3.scale.linear().domain([-0.5 * width, 0.5 * width])
                              .range([0, scale(width)]),
    yscale = d3.scale.linear().domain([-0.5 * height, 0.5 * height])
                              .range([scale(height), 0]);

var svg = d3.select("#canvas").append("svg")
                              .attr("width", scale(width))
                              .attr("height", scale(height));

var star = svg.append("circle")
              .attr("class", "star")
              .attr("cx", xscale(0))
              .attr("cy", yscale(0))
              .attr("r", scale(1));

var bins = d3.range(0.0, 1.1, 0.1);
bins.map(function (r) {
  svg.append("circle").attr("class", "arc")
                      .attr("cx", xscale(0))
                      .attr("cy", yscale(0))
                      .attr("r", scale(r));
});

var planet = svg.append("circle")
              .attr("class", "planet")
              .attr("cx", xscale(b))
              .attr("cy", yscale(0))
              .attr("r", scale(a));

var area_generator = d3.svg.area()
              .x0(function (d) { return xscale(d[1] * Math.cos(d[0])); })
              .x1(function (d) { return xscale(d[2] * Math.cos(d[0])); })
              .y0(function (d) { return yscale(d[1] * Math.sin(d[0])); })
              .y1(function (d) { return yscale(d[2] * Math.sin(d[0])); });
var area_path = svg.append("path").attr("class", "area");

var title = svg.append("text")
               .attr("x", xscale(0))
               .attr("y", yscale(0.5 * height))
               .attr("dy", "1em")
               .attr("text-anchor", "middle")
               .text("a = " + a + "; b = " + b);

var cases = svg.append("text")
               .attr("x", xscale(0))
               .attr("y", yscale(-0.5 * height))
               .attr("dy", "-1em")
               .attr("text-anchor", "middle")
               .text("Cases: ");

function case_number (i) {
  var rn = bins[i + 1], rnm1 = bins[i];
  if (b - a > rn || b + a < rnm1) return 0;
  if (rnm1 + a < b && b <= rn + a) return 1;
  if (Math.sqrt(rn * rn + a * a) < b && b <= rnm1 + a) return 2;
  if (Math.sqrt(a * a + Math.pow(0.5 * (rn + rnm1), 2)) < b
      && b <= Math.sqrt(rn * rn + a * a)) return 3;
  if (Math.sqrt(rnm1 * rnm1 + a * a) < b
      && b <= Math.sqrt(a * a + Math.pow(0.5 * (rn + rnm1), 2))) return 4;
  if (rn - a < b && b <= Math.sqrt(rnm1 * rnm1 + a * a)) return 5;
  if (rnm1 - a < b && b <= rn - a) return 6;
  return -1;
}

function rpm (th) {
  var bsth = b * Math.sin(th),
      x = b * Math.cos(th),
      y = Math.sqrt(a * a - bsth * bsth);
  return [x - y, x + y];
}

var dth = 0.001;
function area (i) {
  var rn = bins[i + 1], rnm1 = bins[i],
      c = case_number(i);
  if (c === 0) return [];
  if (c === 1) {
    var th = d3.range(0, Math.acos((rn*rn+b*b-a*a)/(2*rn*b)), dth),
        r = th.map(function (t) { return [t, rpm(t)[0], rn];});
    return r;
  }
  if (c === 2) {
    var thmin = Math.acos((rnm1*rnm1+b*b-a*a)/(2*rnm1*b)),
        thmax = Math.acos((rn*rn+b*b-a*a)/(2*rn*b)),
        r = d3.range(0, thmin, dth).map(function (t) { return [t, rnm1, rn]; })
              .concat(d3.range(thmin, thmax, dth).map(function (t) {
                return [t, rpm(t)[0], rn];
              }));
    return r;
  }
  if (c === 3) {
    var thmin = Math.acos((rnm1*rnm1+b*b-a*a)/(2*rnm1*b)),
        thmid = Math.acos((rn*rn+b*b-a*a)/(2*rn*b)),
        thmax = Math.asin(a / b),
        r = d3.range(0, thmin, dth).map(function (t) { return [t, rnm1, rn]; })
              .concat(d3.range(thmin, thmid, dth).map(function (t) {
                return [t, rpm(t)[0], rn];
              })).concat(d3.range(thmid, thmax, dth).map(function (t) {
                var rs = rpm(t);
                return [t, rs[0], rs[1]];
              }));
    return r;
  }
  if (c === 4) {
    var thmin = Math.acos((rn*rn+b*b-a*a)/(2*rn*b)),
        thmid = Math.acos((rnm1*rnm1+b*b-a*a)/(2*rnm1*b)),
        thmax = Math.asin(a / b),
        r = d3.range(0, thmin, dth).map(function (t) { return [t, rnm1, rn]; })
              .concat(d3.range(thmin, thmid, dth).map(function (t) {
                return [t, rnm1, rpm(t)[1]];
              })).concat(d3.range(thmid, thmax, dth).map(function (t) {
                var rs = rpm(t);
                return [t, rs[0], rs[1]];
              }));
    return r;
  }
  if (c === 5) {
    var thmin = Math.acos((rn*rn+b*b-a*a)/(2*rn*b)),
        thmax = Math.acos((rnm1*rnm1+b*b-a*a)/(2*rnm1*b)),
        r = d3.range(0, thmin, dth).map(function (t) { return [t, rnm1, rn]; })
              .concat(d3.range(thmin, thmax, dth).map(function (t) {
                return [t, rnm1, rpm(t)[1]];
              }));
    return r;
  }
  if (c === 6) {
    var th = d3.range(0, Math.acos((rnm1*rnm1+b*b-a*a)/(2*rnm1*b)), dth),
        r = th.map(function (t) { return [t, rnm1, rpm(t)[1]];});
    return r;
  }
  return [];
};

function update(a0, b0) {
  a = a0;
  b = b0;
  planet.attr("cx", xscale(b))
        .attr("r", scale(a));
  title.text("a = " + a.toFixed(2) + "; b = " + b.toFixed(2));

  cases.text(bins.slice(1).map(function (r, i) { return case_number(i); }));
  area_path.attr("d", area_generator(area(current_bin)));
}

svg.on("mousemove", function () {
  var c = d3.mouse(this);
  update((yscale.invert(c[1]) + 0.5 * height) / height,
          0.5 * xscale.invert(c[0]) + 0.25 * width);
});

svg.on("click", function () {
  var c = d3.mouse(this),
      x = xscale.invert(c[0]),
      y = yscale.invert(c[1]);
  current_bin = Math.min(Math.floor(Math.sqrt(x * x + y * y) * (bins.length - 1)), bins.length - 2);
});
