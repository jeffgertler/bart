(function () {

  "use strict";

  // Based on Kepler-6b.
  var period = 3.234723,
      a = 7.05,
      t0 = 0.5 * period,
      incl = 86.8,
      p = 0.098;

  var duration = period * (p + 1) / Math.PI / a,
      depth = p * p;

  // Generate a time grid.
  var time = [], ntime = 2000;
  for (var i = 0; i < ntime; ++i)
    time.push(period * i / ntime);

  // Animation stuff.
  var current_frame = 1000;

  // Compute the light curve.
  var ldp = window.Bart.fiducial_ldp(0.3, 0.1),
      lightcurve = new window.Bart.Lightcurve(time, period, a, t0, incl, p, ldp);

  var w = 600, h = 400;
  var svg = d3.select("#plot").append("svg")
                      .attr("width", w)
                      .attr("height", h);
  var line = d3.svg.line().x(function (d) { return d[0]; })
                          .y(function (d) { return d[1]; })

  // Set up the star gradient.
  var defs = svg.append("svg:defs"),
      gradient = defs.append("svg:radialGradient")
                      .attr("id", "gradient")
                      .attr("cx", "50%")
                      .attr("cy", "50%")
                      .attr("fx", "50%")
                      .attr("fy", "50%");
  gradient.append("svg:stop")
      .attr("offset", "0%")
      .attr("stop-color", "#fff")
      .attr("stop-opacity", 1);
  gradient.append("svg:stop")
      .attr("offset", "100%")
      .attr("stop-color", "#f2e5d8")
      .attr("stop-opacity", 0.99);

  // Set up the star drop shadow.
  var drop = defs.append("svg:filter").attr("id", "drop")
                      .attr("x", "-50%")
                      .attr("y", "-50%")
                      .attr("width", "200%")
                      .attr("height", "200%");
  drop.append("svg:feGaussianBlur")
                      .attr("result", "blurOut")
                      .attr("in", "SourceGraphic")
                      .attr("stdDeviation", "10");
  drop.append("svg:feBlend")
                      .attr("in", "SourceGraphic")
                      .attr("in2", "blurOut")
                      .attr("mode", "normal");

  // Plot the top view.
  var top_w = 250, top_h = 250;
  var el_top = svg.append("g")
                      .attr("width", top_w)
                      .attr("height", top_h)
                      .attr("transform", "translate("+(0.5*(0.5*w-top_w))+",0)");

  var da = 1.1 * a,
      xlim = [-da, da],
      ylim = [-da, da],
      xscale = d3.scale.linear().domain(xlim).range([0, top_w]),
      yscale = d3.scale.linear().domain(ylim).range([top_h, 0]);

  var top_data = lightcurve.time.map(function (v, i) {
    return [xscale(lightcurve.pos[i][1]), yscale(-lightcurve.pos[i][0])];
  });

  el_top.append("path").attr("d", line(top_data))
            .attr("class", "track");

  el_top.append("circle")
            .attr("cx", xscale(0))
            .attr("cy", yscale(0))
            .attr("r", top_h/(ylim[1]-ylim[0]))
            .attr("class", "star");

  el_top.append("circle")
                .attr("cx", top_data[current_frame][0])
                .attr("cy", top_data[current_frame][1])
                .attr("r", 2*p*top_h/(ylim[1]-ylim[0]))
                .attr("class", "transit-point");
  svg.append("text")
                .attr("x", 5)
                .attr("y", 15)
                .text("Top view");

  // Plot the side view.
  var side_w = 0.5*w, side_h = top_h;
  var el_side = svg.append("g")
                      .attr("width", side_w)
                      .attr("height", side_h)
                      .attr("transform", "translate("+(0.5*w)+",0)")
                    .append("svg");

  var aspect = side_w / side_h,
      ylim = [-2, 2],
      xlim = [ylim[0] * aspect, ylim[1] * aspect],
      xscale = d3.scale.linear().domain(xlim).range([0, side_w]),
      yscale = d3.scale.linear().domain(ylim).range([side_h, 0]);

  var front_data = [],
      back_data = [],
      side_data = lightcurve.time.map(function (v, i)
  {
    var x = [xscale(lightcurve.pos[i][1]), yscale(-lightcurve.pos[i][2]),
             lightcurve.pos[i][0] < 0 ? 1 : -1];
    if (x[2] < 0) front_data.push(x);
    else back_data.push(x);
    return x;
  });

  var sf = function (a, b) { return a[0] - b[0]; };  // Sorting function.

  el_side.append("path")
            .attr("d", line(back_data.sort(sf)))
            .style("opacity", 0.6)
            .style("z-index", -1)
            .attr("class", "track");

  el_side.append("circle")
            .attr("id", "back-point")
            .attr("cx", side_data[current_frame][0])
            .attr("cy", side_data[current_frame][1])
            .attr("r", p * side_h/(ylim[1]-ylim[0]))
            .attr("class", "side-point");

  el_side.append("circle")
            .attr("cx", xscale(0))
            .attr("cy", yscale(0))
            .attr("r", side_h/(ylim[1]-ylim[0]))
            .style("z-index", 0)
            .attr("class", "star");

  el_side.append("path")
            .attr("d", line(front_data.sort(sf)))
            .style("z-index", 1)
            .attr("class", "track");

  el_side.append("circle")
            .attr("id", "front-point")
            .attr("cx", side_data[current_frame][0])
            .attr("cy", side_data[current_frame][1])
            .attr("r", p * side_h/(ylim[1]-ylim[0]))
            .attr("class", "side-point");

  svg.append("text")
                .attr("text-anchor", "end")
                .attr("x", w - 5)
                .attr("y", 15)
                .text("Observer's point of view");

  // Plot the light curve.
  var transit_w = w, transit_h = h-top_h;
  var el_transit = svg.append("g")
                      .attr("width", transit_w)
                      .attr("height", transit_h);

  var xlim = [0.5 * period - duration, 0.5 * period + duration],
      ylim = [1.0 - 0.75 * depth, 1.0 + 0.25 * depth],
      xscale = d3.scale.linear().domain(xlim).range([0, transit_w]),
      yscale = d3.scale.linear().domain(ylim).range([h, h-transit_h]);

  var transit_data = lightcurve.time.map(function (v, i) {
    return [xscale(v), yscale(lightcurve.flux[i])];
  });

  el_transit.append("path").attr("d", line(transit_data))
                      .attr("class", "track");
  el_transit.selectAll("circle").data([transit_data[current_frame]])
                .enter()
              .append("circle")
                .attr("cx", function(d) { return d[0]; })
                .attr("cy", function(d) { return d[1]; })
                .attr("r", 4)
                .attr("class", "transit-point");
  el_transit.append("text")
                .attr("x", 5)
                .attr("y", h - 8)
                .text("Observed brightness as a function of time");

  // Animation shim from:
  //    http://paulirish.com/2011/requestanimationframe-for-smart-animating/
  window.requestAnimFrame = (function(){
    return  window.requestAnimationFrame       ||
            window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame    ||
            window.oRequestAnimationFrame      ||
            window.msRequestAnimationFrame     ||
            function(callback){
              window.setTimeout(callback, 1000 / 60);
            };
  })();

  // Animate.
  function animate () {
    requestAnimFrame(animate);
    current_frame = (current_frame + 2) % ntime;
    el_top.selectAll("circle.transit-point").data([top_data[current_frame]])
      .attr("cx", function(d) { return d[0]; })
      .attr("cy", function(d) { return d[1]; });

    if (side_data[current_frame][2] < 0) {
      el_side.selectAll("circle#front-point").data([side_data[current_frame]])
        .attr("cx", function(d) { return d[0]; })
        .attr("cy", function(d) { return d[1]; });
    } else {
      el_side.selectAll("circle#back-point").data([side_data[current_frame]])
        .attr("cx", function(d) { return d[0]; })
        .attr("cy", function(d) { return d[1]; });
    }

    el_transit.selectAll("circle").data([transit_data[current_frame]])
      .attr("cx", function(d) { return d[0]; })
      .attr("cy", function(d) { return d[1]; });
  }
  animate();

})();
