// ========================================================================
//                                                          PHYSICAL SYSTEM
// ========================================================================

var Lightcurve = function () {
  this.p = 0.3;
  this.amin = 0.05;
  this.amax = 1.05;
  this.orbital_radius = 5.0;
  this.th = 0.0;
  this.dth = 0.08;
  this.incl = 0.0;
  this.max_incl = 45.0 * Math.PI / 180.0;

  // The limb-darkening profile.
  this.ldp = d3.range(0.1, 1.1, 0.3).map(function (r0) {
    // var r = Math.pow(r0, 0.5),
    //     onemmu = 1 - Math.sqrt(1 - r);
    // return [r, 1 - 0.5 * onemmu - 0.1 * onemmu * onemmu];
    return [r0, 1];
  });

  // The 3D element.
  this.three = setup_3d(this);

  // The transit plot.
  this.pl = plot3().width(700)
                   .height(300)
                   .margin({top: 20, right: 10, bottom: 30, left:50})
                   .xlabel("Phase")
                   .ylabel("Relative Flux")
                   .ylim([0.85, 1.01]);
  this.pl.plot("time", "flux", {label: "data"});
  this.pl.plot("time", "flux", {label: "current", render: "scatter"});

  // The limb darkening plot.
  this.ldpl = plot3().width(400)
                   .height(250)
                   .margin({top: 20, right: 10, bottom: 30, left:50})
                   .xlabel("r")
                   .xlim([0, 1])
                   .ylim([0, 1]);
  this.ldpl.plot(function (d) { return d[0]; }, function (d) { return d[1]; });
  this.ldpl.plot(function (d) { return d[0]; }, function (d) { return d[1]; },
               {"render": "scatter"});
};

Lightcurve.prototype.bind = function (three, two, limb) {
  this.three(three);
  this.two = two;
  this.limb = limb;
  this.redraw();
}

Lightcurve.prototype.advance = function () {
  this.th += this.dth;
  this.three.move_planet();
  this.three.render();

  var th = (this.th + Math.PI) % (2 * Math.PI) - Math.PI;
  this.pl(d3.select(this.two).datum({data: this.data,
                                     current: [{time: th,
                                                flux: this.flux(th)}]}));

  this.ldpl(d3.select(this.limb).datum([[0, 1]].concat(this.ldp)));
};

Lightcurve.prototype.redraw = function () {
  this.three.update_camera();
  this.data = this.compute();
};

Lightcurve.prototype.flux = function (t) {
  var x = this.orbital_radius * Math.cos(t),
      y = this.orbital_radius * Math.sin(t),
      x0 = x * Math.cos(this.incl),
      z0 = x * Math.sin(this.incl),
      b = Math.sqrt(y * y + z0 * z0),
      p = this.p;

  if (x0 < 0 || b + p > 1) return 1.0;

  var areas = this.ldp.map(function (d) {
    return compute_occultation(d[0], p, b);
  });

  console.log(b, areas);

  var ldp = this.ldp;
  var occulted_flux = areas[0] + areas.slice(1).reduce(function (curr, a, i) {
    return curr + ldp[i][1] * (a - areas[i]);
  }, 0.0);

  return 1 - occulted_flux / this.total_flux;
};

Lightcurve.prototype.compute = function () {
  var th = d3.range(-Math.PI, Math.PI, 0.001);

  // Compute the total flux from the star for normalization purposes.
  this.total_flux = Math.PI * this.ldp.reduce(function (curr, d, i, arr) {
    if (i === 0) return curr + d[1] * d[0] * d[0];
    return curr + d[1] * d[0] * d[0] - d[1] * arr[i - 1][0] * arr[i - 1][0];
  }, 0.0);

  return th.map(function (t) {
    return {time: t, flux: this.flux(t)};
  }, this);

  throw "sup";
};

function compute_occultation (r0, p, b) {
  if (b >= r0 + p) return 0.0;
  if (b <= p - r0) return Math.PI * r0 * r0;
  if (b <= r0 - p) return Math.PI * p * p;

  var r2 = r0 * r0,
      p2 = p * p,
      b2 = b * b,
      k0 = Math.acos((b2 + p2 - r2) / (2 * b * p)),
      k1 = Math.acos((b2 + r2 - p2) / (2 * b * r0)),
      k2 = Math.sqrt((p + r0 - b) * (b + p - r0) * (b - p + r0) * (b + r0 + p));

  return p2 * k0 + r2 * k1 - 0.5 * k2;
}


// ========================================================================
//                                                         3D VISUALIZATION
// ========================================================================

function setup_3d (lc_) {
  var lc = lc_;

  // Set up the scene and viewport.
  var width = 700, height = 400, view_angle = 1.0, aspect = width / height,
      camera_dist = 20000,
      near = camera_dist - 5000, far = camera_dist + 5000,
      rs = 50, $el;

  var renderer = new THREE.WebGLRenderer(),
      camera = new THREE.PerspectiveCamera(view_angle, aspect, near, far),
      scene = new THREE.Scene();

  var star, planet;

  var three_dee = function (el) {
    scene.add(camera);
    renderer.setSize(width, height);

    $el = $(el);
    $el.attr("width", width);
    $el.attr("height", height);
    $el.append(renderer.domElement);

    // Lighting.
    var light = new THREE.PointLight(0xFFFFFF);

    light.position.x = 10;
    light.position.y = 50;
    light.position.z = 1000;
    scene.add(light);

    // Set up the star.
    var seg = 24, rings = 24,
        star_mat = new THREE.MeshLambertMaterial({color: 0xFFC40D});

    star = new THREE.Mesh(new THREE.SphereGeometry(rs, seg, rings), star_mat);
    scene.add(star);

    // Set up the planet.
    var radp = lc.p * rs, segp = 16, ringsp = 16,
        planet_mat = new THREE.MeshLambertMaterial({color: 0x00EEAA});

    planet = new THREE.Mesh(new THREE.SphereGeometry(radp, segp, ringsp), planet_mat);
    scene.add(planet);

    // Draw the orbit.
    var geom = new THREE.Geometry();
    d3.range(0, 2 * Math.PI, 0.01).forEach(function (t) {
      geom.vertices.push(new THREE.Vector3(rs * lc.orbital_radius * Math.cos(t),
          0, rs * lc.orbital_radius * Math.sin(t)));
    });

    var orbit_mat = new THREE.LineBasicMaterial({color: 0xBBBBBB}),
        orbit = new THREE.Line(geom, orbit_mat);
    scene.add(orbit);

    // Events.
    var moving = false;
    $el.on("mousemove", function (e) {
      if (!moving) return false;
      var y = e.pageY - this.offsetTop;
      lc.incl = lc.max_incl * (y / height - 0.5);
      lc.redraw();
    });

    $el.on("mouseout", function (e) { moving = false; });
    $el.on("mouseup", function (e) { moving = false; });
    $el.on("mousedown", function (e) { moving = true; });

    return three_dee;
  };

  three_dee.update_camera = function () {
    camera.position.z = camera_dist * Math.cos(lc.incl);
    camera.position.y = camera_dist * Math.sin(lc.incl);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
  };

  three_dee.move_planet = function () {
    planet.position.z = lc.orbital_radius * rs * Math.cos(lc.th);
    planet.position.x = lc.orbital_radius * rs * Math.sin(lc.th);
  }

  three_dee.render = function () {
    renderer.render(scene, camera);
  }

  return three_dee;
}


// ========================================================================
//                                                                ANIMATION
// ========================================================================

$(function () {

  var lc = new Lightcurve();
  lc.bind("#three-dee", "#two-dee", "#limb-darkening");

  function animate () {
    requestAnimationFrame(animate);
    lc.advance();
  }

  animate();

});
