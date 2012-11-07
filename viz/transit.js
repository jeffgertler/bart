// ========================================================================
//                                                          PHYSICAL SYSTEM
// ========================================================================

var Lightcurve = function () {
  this.a = 0.3;
  this.amin = 0.05;
  this.amax = 1.05;
  this.orbital_radius = 5.0;
  this.th = 0.0;
  this.dth = 0.08;
  this.incl = 0.0;
  this.max_incl = 45.0 * Math.PI / 180.0;

  this.three = setup_3d(this);

  this.pl = plot3().width(700)
                   .height(300)
                   .margin({top: 20, right: 10, bottom: 30, left:50})
                   .xlabel("Phase")
                   .ylabel("Relative Flux")
                   .ylim([0.85, 1.01]);

  this.pl.plot("time", "flux", {label: "data"});
  this.pl.plot("time", "flux", {label: "current", render: "scatter"});
};

Lightcurve.prototype.bind = function (three, two) {
  this.three(three);
  this.two = two;
  this.redraw();
}

Lightcurve.prototype.advance = function () {
  this.th += this.dth;
  this.three.move_planet();
  this.three.render();

  var th = (this.th + Math.PI) % (2 * Math.PI) - Math.PI;
  this.pl(d3.select(this.two).datum({data: this.data,
                                     current: [{time: th,
                                                flux: this.occultation(th)}]}));
};

Lightcurve.prototype.redraw = function () {
  this.three.update_camera();
  this.data = this.compute();
};

Lightcurve.prototype.occultation = function (t) {
  var x = this.orbital_radius * Math.cos(t),
      y = this.orbital_radius * Math.sin(t),
      x0 = x * Math.cos(this.incl),
      z0 = x * Math.sin(this.incl),
      b = Math.sqrt(y * y + z0 * z0),
      a = this.a;

  if (x0 < 0 || b >= 1 + a) return 1.0;
  else if (Math.abs(1 - a) < b  && b < 1 + a) {
    var a2 = a * a,
      b2 = b * b,
         tmp = b2 - a2 + 1,
         k1 = Math.acos(0.5 * tmp / b),
         k2 = Math.acos(0.5 * (b2 + a2 - 1) / b / a),
         k3 = Math.sqrt(b2 - 0.25 * tmp * tmp);
    return 1 - (k1 + a2 * k2 - k3) / Math.PI;
  } else if (b <= 1 - a) return 1 - a * a;
  return 0.0;
};

Lightcurve.prototype.compute = function () {
  var th = d3.range(-Math.PI, Math.PI, 0.001);

  return th.map(function (t) {
    return {time: t, flux: this.occultation(t)};
  }, this);
};


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
    var radp = lc.a * rs, segp = 16, ringsp = 16,
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
  lc.bind("#three-dee", "#two-dee");

  function animate () {
    requestAnimationFrame(animate);
    lc.advance();
  }

  animate();

});
