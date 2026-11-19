// Planet Finders galaxy UI.
// Galaxy view: galactocentric coordinates in parsecs, z up (galactic north), Sun near (-8122, 0, 21).
// System view: separate scene in units of the host star's radius.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x02030a);
const PX = renderer.getPixelRatio();

const clock = new THREE.Clock();
let mode = "galaxy"; // galaxy | swoop | system | return
let DATA = null;

// ---------------------------------------------------------------- shared helpers
function mouseControls(camera, el) {
  const c = new OrbitControls(camera, el);
  c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
  c.screenSpacePanning = true;
  c.enableDamping = true;
  c.dampingFactor = 0.08;
  c.zoomToCursor = true;
  c.rotateSpeed = 0.6;
  return c;
}

function rand(seed) { // deterministic galaxy
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}
const R = rand(7);
const gauss = () => { let u = 0, v = 0; while (u === 0) u = R(); v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function teffColor(t) { // approximate blackbody colour
  t = clamp(t || 5600, 2500, 12000) / 100;
  let r, g, b;
  r = t <= 66 ? 255 : 329.7 * Math.pow(t - 60, -0.1332);
  g = t <= 66 ? 99.47 * Math.log(t) - 161.12 : 288.12 * Math.pow(t - 60, -0.0755);
  b = t >= 66 ? 255 : t <= 19 ? 0 : 138.52 * Math.log(t - 10) - 305.04;
  return new THREE.Color(clamp(r, 0, 255) / 255, clamp(g, 0, 255) / 255, clamp(b, 0, 255) / 255);
}

function glowTexture(inner = "rgba(255,255,255,1)", mid = "rgba(255,255,255,0.25)") {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, inner); grd.addColorStop(0.2, mid); grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function starfield(n, radius, size) {
  const p = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = R() * 2 - 1, th = R() * Math.PI * 2, s = Math.sqrt(1 - u * u);
    p.set([radius * s * Math.cos(th), radius * s * Math.sin(th), radius * u], i * 3);
    const c = teffColor(3000 + R() * 9000).multiplyScalar(0.35 + R() * 0.65);
    col.set([c.r, c.g, c.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: true,
    transparent: true, opacity: 0.8, depthWrite: false }));
}

// ---------------------------------------------------------------- galaxy scene
const gScene = new THREE.Scene();
const gCam = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.01, 2e6);
gCam.up.set(0, 0, 1);
gCam.position.set(4000, -30000, 21000);
const gCtl = mouseControls(gCam, canvas);
gCtl.target.set(-1500, 0, 0);
gCtl.minDistance = 0.05;
gCtl.maxDistance = 150000;
// OrbitControls zooms by a fraction of the distance to its target, so each scroll step shrinks as you
// close in and you stall before reaching anything. When zooming in closer than this floor, slide the
// target forward along the view so the step size stays constant and you fly on through the stars.
const ZOOM_FLOOR_PC = 150;
canvas.addEventListener("wheel", (e) => {
  if (mode !== "galaxy" || e.deltaY >= 0) return;
  const d = gCam.position.distanceTo(gCtl.target);
  if (d >= ZOOM_FLOOR_PC) return;
  const fwd = new THREE.Vector3(); gCam.getWorldDirection(fwd);
  gCtl.target.copy(gCam.position).addScaledVector(fwd, ZOOM_FLOOR_PC);
}, { capture: true, passive: true });
gScene.add(starfield(6000, 9e5, 1.2));

const GALAXY_VERT = `
    attribute float aSize; attribute vec3 aColor; attribute float aAlpha;
    uniform float uScale; uniform float uMax; varying vec3 vC; varying float vA;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float d = -mv.z;
      float s = aSize * uScale / d;
      float a = aAlpha * smoothstep(40.0, 600.0, d);    // fade out when we fly through the disk
      if (s < 1.0) { a *= s; s = 1.0; }
      if (s > uMax) { a *= uMax / s; s = uMax; }         // up close: stay small and faint instead of turning into fog
      gl_PointSize = s;
      vC = aColor; vA = a;
      gl_Position = projectionMatrix * mv;
    }`;
// Stars: additive, tight falloff so the disk reads as many faint points rather than a glow.
const galaxyMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: { uScale: { value: 1 }, uMax: { value: 14 * PX } },
  vertexShader: GALAXY_VERT,
  fragmentShader: `
    varying vec3 vC; varying float vA;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      float f = exp(-r * r * 7.0);
      if (f < 0.02) discard;
      gl_FragColor = vec4(vC * f * vA, 1.0);
    }`,
});
// Dust: normal blending with dark colour, drawn over the stars to carve lanes along the arms.
const dustMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.NormalBlending,
  uniforms: galaxyMat.uniforms,
  vertexShader: GALAXY_VERT,
  fragmentShader: `
    varying vec3 vC; varying float vA;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      float f = (1.0 - smoothstep(0.2, 1.0, r));
      if (f < 0.01) discard;
      gl_FragColor = vec4(vC, f * vA);
    }`,
});

function pointCloud(n, fill, mat, order) {
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n), alpha = new Float32Array(n);
  let i = 0;
  const put = (x, y, z, c, s, a) => { pos.set([x, y, z], i * 3); col.set([c.r, c.g, c.b], i * 3); size[i] = s; alpha[i] = a; i++; };
  while (i < n) fill(put);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
  g.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
  g.setAttribute("aAlpha", new THREE.BufferAttribute(alpha, 1));
  const pts = new THREE.Points(g, mat); pts.renderOrder = order;
  gScene.add(pts);
}

function buildGalaxy() {
  const young = new THREE.Color(0.62, 0.74, 1.0), white = new THREE.Color(0.9, 0.9, 0.95), old = new THREE.Color(1.0, 0.82, 0.6);
  const disk = new THREE.Color(0.78, 0.72, 0.66), pink = new THREE.Color(1.0, 0.42, 0.62), dustC = new THREE.Color(0.035, 0.022, 0.02);
  const pitch = (12.5 * Math.PI) / 180, k = 1 / Math.tan(pitch), r0 = 3200;
  const arms = [0, Math.PI / 2, Math.PI, 1.5 * Math.PI], armW = [1, 0.5, 1, 0.5]; // two major, two minor arms
  const barAng = (-27 * Math.PI) / 180;
  const armPoint = (spread, offset = 0) => {
    let a; do { a = Math.floor(R() * 4); } while (R() > armW[a]);
    const r = 2600 + -Math.log(1 - R() * 0.985) * 3800;
    if (r > 16500) return null;
    const th = arms[a] + Math.log(r / r0) * k + offset + gauss() * spread * (1 + 1500 / r);
    const rr = r * (1 + gauss() * 0.035);
    return [rr * Math.cos(th), rr * Math.sin(th), r];
  };
  const bar = () => { const u = gauss() * 1900, v = gauss() * 520; return [u * Math.cos(barAng) - v * Math.sin(barAng), u * Math.sin(barAng) + v * Math.cos(barAng)]; };

  // smooth old-star disk: the faint glow between the arms
  pointCloud(60000, (put) => {
    const r = -Math.log(1 - R() * 0.99) * 3600; if (r > 17000) return;
    const t = R() * Math.PI * 2;
    put(r * Math.cos(t), r * Math.sin(t), gauss() * (180 + r * 0.01), disk.clone().lerp(old, clamp(1 - r / 8000, 0, 1)), 80, 0.07);
  }, galaxyMat, 0);
  // bulge + bar
  pointCloud(32000, (put) => {
    let x, y, z;
    if (R() < 0.5) { [x, y] = bar(); z = gauss() * 300; }
    else { const r = Math.abs(gauss()) * 1100, t = R() * Math.PI * 2, u = R() * 2 - 1; x = r * Math.sqrt(1 - u * u) * Math.cos(t); y = r * Math.sqrt(1 - u * u) * Math.sin(t); z = r * u * 0.6; }
    put(x, y, z, old.clone().offsetHSL(0, 0, (R() - 0.5) * 0.12), 70 + R() * 50, 0.11 + R() * 0.07);
  }, galaxyMat, 0);
  // spiral arms: many small, faint, slightly varied stars
  pointCloud(110000, (put) => {
    const p = armPoint(0.15); if (!p) return;
    const [x, y, r] = p;
    const c = young.clone().lerp(white, R() * 0.8).lerp(old, clamp(1 - r / 6000, 0, 1) * 0.6);
    const bright = R() < 0.04;
    put(x, y, gauss() * (110 + r * 0.01), c, bright ? 60 : 32 + R() * 28, bright ? 0.4 : 0.15 + R() * 0.13);
  }, galaxyMat, 0);
  // pink star-forming knots, sparse
  pointCloud(1200, (put) => {
    const p = armPoint(0.07); if (!p || p[2] < 4000) return;
    put(p[0], p[1], gauss() * 50, pink, 40 + R() * 40, 0.4);
  }, galaxyMat, 0);
  // dust lanes on the inner edge of each arm and along the bar
  pointCloud(26000, (put) => {
    if (R() < 0.12) { const [x, y] = bar(); put(x * 1.1, y * 1.1 + gauss() * 150, gauss() * 60, dustC, 160 + R() * 120, 0.22); return; }
    const p = armPoint(0.06, -0.09); if (!p || p[2] > 13000) return;
    put(p[0], p[1], gauss() * 70, dustC, 120 + R() * 140, 0.16 + R() * 0.1);
  }, dustMat, 1);

  const coreGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,215,160,0.5)", "rgba(255,170,90,0.10)"),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  coreGlow.scale.set(5200, 5200, 1);
  gScene.add(coreGlow);
}

// Kepler stars: every placed star dim, positives bright purple.
const pointMat = (vertex, fragment, uniforms) => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms, vertexShader: vertex, fragmentShader: fragment });

const dimMat = pointMat(`
    uniform float uPx;
    varying float vA;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float d = -mv.z;
      vA = clamp(900.0 / d, 0.12, 1.0);
      gl_PointSize = uPx * mix(1.4, 3.2, clamp(60.0 / d, 0.0, 1.0));
      gl_Position = projectionMatrix * mv;
    }`, `
    varying float vA;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      if (r > 1.0) discard;
      gl_FragColor = vec4(vec3(0.62, 0.74, 1.0) * (1.0 - r * r) * vA, 1.0);
    }`, { uPx: { value: PX } });

const makePosMat = (blending) => { const m = pointMat(`
    attribute float aP; attribute float aId; attribute float aTest;
    uniform float uPx; uniform float uTime; uniform float uHover; uniform float uShowTest; uniform float uShowSeen;
    varying float vP; varying float vH; varying float vI; varying float vT;
    void main() {
      vT = aTest;
      if ((aTest > 0.5 ? uShowTest : uShowSeen) < 0.5) { gl_PointSize = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float d = -mv.z;
      float pulse = 1.0 + 0.18 * sin(uTime * 2.2 + aId * 1.7);
      vH = abs(aId - uHover) < 0.5 ? 1.0 : 0.0;
      float near = clamp(40.0 / d, 1.0, 2.6);
      float far = clamp(2500.0 / d, 0.16, 1.0);   // thousands overlap when seen from afar: dim them so the cluster keeps its colour
      vI = max(far, vH);
      gl_PointSize = uPx * (5.0 + 5.0 * aP) * pulse * near * mix(0.6, 1.0, far) * (1.0 + vH * 0.9);
      vP = aP;
      gl_Position = projectionMatrix * mv;
    }`, `
    varying float vP; varying float vH; varying float vI; varying float vT;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      if (r > 1.0) discard;
      vec3 hue = vT > 0.5 ? vec3(0.2, 1.0, 0.32) : vec3(0.74, 0.25, 1.0);   // neon green = held-out test, purple = seen in training
      vec3 hot = vT > 0.5 ? vec3(0.9, 1.0, 0.9) : vec3(1.0, 0.92, 1.0);
      float core = exp(-r * r * 30.0) * vI * vI;
      float halo = exp(-r * r * 5.0) * (0.55 + 0.45 * vP) * vI * (vT > 0.5 ? 1.25 : 1.0);
      vec3 c = hue * halo + hot * core * (vT > 0.5 ? 1.0 : 0.45) + vec3(0.3, 0.9, 1.0) * vH * halo * 0.6;
      gl_FragColor = vec4(c, 1.0);
      #ifdef OPAQUE_DOTS
        // normal blend keeps the hue: saturated neon rim, hot centre, soft edge
        gl_FragColor = vec4(mix(vec3(0.15, 1.0, 0.3), vec3(0.85, 1.0, 0.85), exp(-r * r * 18.0)), smoothstep(0.85, 0.35, r));
      #endif
    }`, posUniforms); m.blending = blending; return m; };
const posUniforms = { uPx: { value: PX }, uTime: { value: 0 }, uHover: { value: -1 }, uShowTest: { value: 1 }, uShowSeen: { value: 1 }, uShowDis: { value: 1 } };
const posMat = makePosMat(THREE.AdditiveBlending);          // train/val: additive glow
const testMat = makePosMat(THREE.NormalBlending);            // held-out: solid green dots drawn on top
testMat.defines = { OPAQUE_DOTS: 1 };
testMat.transparent = true;

// Gold reticle around stars where the model says "planet" but the catalog still says "candidate".
// Only held-out stars (test, val) count: on training stars the model just echoes its labels.
const disMat = pointMat(`
    attribute float aTest; attribute float aId;
    uniform float uPx; uniform float uHover; uniform float uShowTest; uniform float uShowSeen; uniform float uShowDis;
    varying float vH;
    void main() {
      if (uShowDis < 0.5 || (aTest > 0.5 ? uShowTest : uShowSeen) < 0.5) { gl_PointSize = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float d = -mv.z;
      vH = abs(aId - uHover) < 0.5 ? 1.0 : 0.0;
      gl_PointSize = uPx * 30.0 * clamp(40.0 / d, 1.0, 2.0) * mix(0.75, 1.0, clamp(2500.0 / d, 0.0, 1.0)) * (1.0 + vH * 0.5);
      gl_Position = projectionMatrix * mv;
    }`, `
    uniform float uTime;
    varying float vH;
    void main() {
      vec2 q = gl_PointCoord - 0.5;
      float r = length(q) * 2.0;
      float a = atan(q.y, q.x) + uTime * 0.6;
      float gaps = step(0.28, abs(fract(a / 1.5708) - 0.5) * 2.0);   // four slow-turning brackets
      float ring = smoothstep(0.1, 0.04, abs(r - 0.78)) * gaps;
      float edge = smoothstep(0.2, 0.1, abs(r - 0.78)) * gaps;        // dark outline so it reads over the crowd
      float alpha = max(ring, edge * 0.75);
      if (alpha < 0.02) discard;
      vec3 gold = mix(vec3(1.0, 0.76, 0.2), vec3(1.0, 0.96, 0.75), vH);
      gl_FragColor = vec4(gold * ring / max(alpha, 1e-3), alpha);
    }`, posUniforms);
disMat.blending = THREE.NormalBlending;

// model says planet, catalog (current KOI table) still says candidate
const isDisagree = (s) => s.label === "PC" && s.pred === "planet" && s.koi_disposition === "CANDIDATE";
const disagreeHeld = (st) => st.signals.some((s) => s.split !== "train" && isDisagree(s));

let posPositions = null; // Float32Array of positive star positions, for picking
let posTest = null;      // Uint8Array: 1 = held-out test star
const isTestStar = (st) => st.signals[0].split === "test"; // splits are star-level
const pickable = (i) => (posTest[i] ? posMat.uniforms.uShowTest.value : posMat.uniforms.uShowSeen.value) > 0.5;

function buildKepler() {
  const s = DATA.stars, n = s.kepid.length;
  const p = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) p.set([s.x[i], s.y[i], s.z[i]], i * 3);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  const dim = new THREE.Points(g, dimMat); dim.renderOrder = 2; // after the dust, so dust never hides them
  gScene.add(dim);

  const P = DATA.positives, m = P.length;
  posPositions = new Float32Array(m * 3);
  const pr = new Float32Array(m), id = new Float32Array(m), te = new Float32Array(m);
  posTest = new Uint8Array(m);
  P.forEach((st, i) => { posPositions.set([st.x, st.y, st.z], i * 3); pr[i] = st.p_planet; id[i] = i; te[i] = posTest[i] = isTestStar(st) ? 1 : 0; });
  const g2 = new THREE.BufferGeometry();
  g2.setAttribute("position", new THREE.BufferAttribute(posPositions, 3));
  g2.setAttribute("aP", new THREE.BufferAttribute(pr, 1));
  g2.setAttribute("aId", new THREE.BufferAttribute(id, 1));
  g2.setAttribute("aTest", new THREE.BufferAttribute(te, 1));
  const pts = new THREE.Points(g2, posMat); pts.renderOrder = 3;
  gScene.add(pts);
  // second pass: only the test stars, solid, on top of the purple glow
  const ti = [...posTest.keys()].filter((i) => posTest[i]);
  const g3 = new THREE.BufferGeometry();
  const pick = (src, k) => { const a = new Float32Array(ti.length * k); ti.forEach((i, j) => a.set(src.subarray(i * k, i * k + k), j * k)); return a; };
  g3.setAttribute("position", new THREE.BufferAttribute(pick(posPositions, 3), 3));
  g3.setAttribute("aP", new THREE.BufferAttribute(pick(pr, 1), 1));
  g3.setAttribute("aId", new THREE.BufferAttribute(pick(id, 1), 1));
  g3.setAttribute("aTest", new THREE.BufferAttribute(pick(te, 1), 1));
  const tp = new THREE.Points(g3, testMat); tp.renderOrder = 4;
  gScene.add(tp);
  // third pass: gold reticles on held-out stars where the model disagrees with the catalog
  const di = P.map((st, i) => (disagreeHeld(st) ? i : -1)).filter((i) => i >= 0);
  const pick2 = (src, k) => { const a = new Float32Array(di.length * k); di.forEach((i, j) => a.set(src.subarray(i * k, i * k + k), j * k)); return a; };
  const g4 = new THREE.BufferGeometry();
  g4.setAttribute("position", new THREE.BufferAttribute(pick2(posPositions, 3), 3));
  g4.setAttribute("aId", new THREE.BufferAttribute(pick2(id, 1), 1));
  g4.setAttribute("aTest", new THREE.BufferAttribute(pick2(te, 1), 1));
  const dp = new THREE.Points(g4, disMat); dp.renderOrder = 5;
  gScene.add(dp);
  $("n-dis").textContent = di.length.toLocaleString();

  const sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,240,200,1)", "rgba(255,190,60,0.5)"),
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: false }));
  sun.scale.set(0.022, 0.022, 1);
  sun.position.set(DATA.sun.x, DATA.sun.y, DATA.sun.z);
  gScene.add(sun);
  const nTest = posTest.reduce((a, b) => a + b, 0);
  $("n-test").textContent = nTest.toLocaleString();
  $("n-pos").textContent = (m - nTest).toLocaleString();
  for (const [id, key] of [["show-test", "uShowTest"], ["show-seen", "uShowSeen"], ["show-dis", "uShowDis"]]) {
    $(id).onchange = (e) => { posMat.uniforms[key].value = e.target.checked ? 1 : 0; };
  }
  $("n-all").textContent = n.toLocaleString();
}

// HTML labels that follow 3-D points
const labels = [];
function addLabel(text, v, cls = "") {
  const el = document.createElement("div"); el.className = "label " + cls; el.textContent = text;
  $("labels").appendChild(el); labels.push({ el, v });
}
function updateLabels(cam) {
  const show = mode === "galaxy" || mode === "swoop" || mode === "return";
  for (const l of labels) {
    const p = l.v.clone().project(cam);
    const vis = show && p.z < 1 && Math.abs(p.x) < 1.1 && Math.abs(p.y) < 1.1;
    l.el.style.display = vis ? "block" : "none";
    if (vis) { l.el.style.left = ((p.x + 1) / 2) * innerWidth + "px"; l.el.style.top = ((1 - p.y) / 2) * innerHeight + "px"; }
  }
}

// ---------------------------------------------------------------- picking + hover
const tooltip = $("tooltip");
let hoverIdx = -1;
function pickPositive(cx, cy, radiusPx = 14) {
  if (!posPositions) return -1;
  const v = new THREE.Vector3(); let best = -1, bestD = radiusPx * radiusPx;
  const camPos = gCam.position; let bestCam = Infinity;
  for (let i = 0; i < posPositions.length / 3; i++) {
    if (!pickable(i)) continue;
    v.set(posPositions[i * 3], posPositions[i * 3 + 1], posPositions[i * 3 + 2]);
    const dc = v.distanceToSquared(camPos);
    v.project(gCam);
    if (v.z > 1 || v.z < -1) continue;
    const sx = ((v.x + 1) / 2) * innerWidth - cx, sy = ((1 - v.y) / 2) * innerHeight - cy, d = sx * sx + sy * sy;
    if (d < bestD - 1e-6 || (Math.abs(d - bestD) < 9 && dc < bestCam)) { best = i; bestD = d; bestCam = dc; }
  }
  return best;
}
function starName(st) {
  const named = st.signals.find((s) => s.name);
  return named ? named.name.replace(/ [a-z]$/, "") : `KIC ${st.kepid}`;
}
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);

let down = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, b: e.button }; canvas.classList.add("dragging"); });
canvas.addEventListener("pointerup", (e) => {
  canvas.classList.remove("dragging");
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  if (down.b === 0 && moved < 5) {
    if (mode === "galaxy") { const i = pickPositive(e.clientX, e.clientY); if (i >= 0) swoopTo(i); }
    else if (mode === "system") pickPlanet(e.clientX, e.clientY);
  }
  down = null;
});
canvas.addEventListener("pointermove", (e) => {
  if (mode !== "galaxy" || down) { tooltip.style.display = "none"; return; }
  hoverIdx = pickPositive(e.clientX, e.clientY);
  posMat.uniforms.uHover.value = hoverIdx;
  canvas.classList.toggle("pointer", hoverIdx >= 0);
  if (hoverIdx >= 0) {
    const st = DATA.positives[hoverIdx];
    const test = posTest[hoverIdx];
    tooltip.innerHTML = `${starName(st)} <span class="${test ? "t" : "s"}">${test ? "HELD-OUT" : st.signals[0].split.toUpperCase()}</span>${disagreeHeld(st) ? ` <span class="d">MODEL: PLANET · CATALOG: CANDIDATE</span>` : ""}<br><span class="p">planet probability ${pct(st.p_planet)}</span> · ${fmtDist(st.dist_pc)}`;
    tooltip.style.display = "block";
    tooltip.style.left = e.clientX + 14 + "px"; tooltip.style.top = e.clientY + 12 + "px";
  } else tooltip.style.display = "none";
});
const fmtDist = (d) => (d == null ? "?" : d >= 1000 ? `${(d / 1000).toFixed(2)} kpc` : `${Math.round(d)} pc`);

// ---------------------------------------------------------------- swoop
let swoop = null, saved = null;
function swoopTo(i) {
  const st = DATA.positives[i];
  const S = new THREE.Vector3(st.x, st.y, st.z);
  const P0 = gCam.position.clone(), T0 = gCtl.target.clone();
  saved = { pos: P0.clone(), target: T0.clone() };
  const dist = P0.distanceTo(S);
  const away = P0.clone().sub(S).normalize();
  const up = new THREE.Vector3(0, 0, 1);
  const endDir = away.clone().lerp(up, 0.35).normalize();
  const E = S.clone().add(endDir.multiplyScalar(0.6));
  const side = new THREE.Vector3().crossVectors(away, up).normalize();
  if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
  const C = P0.clone().lerp(E, 0.55).add(side.multiplyScalar(dist * 0.28)).add(up.clone().multiplyScalar(dist * 0.18));
  swoop = { i, S, P0, T0, C, E, t0: clock.elapsedTime, dur: clamp(1.6 + 0.45 * Math.log10(1 + dist), 2.0, 4.2), flashed: false };
  mode = "swoop"; gCtl.enabled = false; tooltip.style.display = "none"; $("help").style.opacity = 0;
}
function stepSwoop() {
  const s = swoop, t = clamp((clock.elapsedTime - s.t0) / s.dur, 0, 1), e = ease(t);
  const a = s.P0.clone().multiplyScalar((1 - e) ** 2), b = s.C.clone().multiplyScalar(2 * (1 - e) * e), c = s.E.clone().multiplyScalar(e * e);
  gCam.position.copy(a.add(b).add(c));
  gCtl.target.copy(s.T0.clone().lerp(s.S, ease(clamp(t * 1.6, 0, 1))));
  gCam.fov = 55 + 18 * Math.sin(Math.PI * e);
  gCam.updateProjectionMatrix();
  gCam.lookAt(gCtl.target);
  if (t > 0.82 && !s.flashed) { s.flashed = true; flash(); }
  if (t >= 1) { gCam.fov = 55; gCam.updateProjectionMatrix(); enterSystem(s.i); }
}
function flash() {
  $("flash").animate([{ opacity: 0 }, { opacity: 1, offset: 0.35 }, { opacity: 0 }], { duration: 1100, easing: "ease-out" });
}
function startReturn() {
  if (mode !== "system") return;
  flash();
  hideHolo();
  mode = "return";
  const S = sys.galaxyPos;
  swoop = { S, P0: S.clone().add(new THREE.Vector3(0.4, -0.4, 0.3)), T0: S.clone(), E: saved.pos, C: null, t0: clock.elapsedTime, dur: 2.4, target: saved.target };
  swoop.C = swoop.P0.clone().lerp(swoop.E, 0.5).add(new THREE.Vector3(0, 0, swoop.P0.distanceTo(swoop.E) * 0.2));
  disposeSystem();
}
function stepReturn() {
  const s = swoop, t = clamp((clock.elapsedTime - s.t0) / s.dur, 0, 1), e = ease(t);
  const a = s.P0.clone().multiplyScalar((1 - e) ** 2), b = s.C.clone().multiplyScalar(2 * (1 - e) * e), c = s.E.clone().multiplyScalar(e * e);
  gCam.position.copy(a.add(b).add(c));
  gCtl.target.copy(s.S.clone().lerp(s.target, ease(clamp((t - 0.2) / 0.8, 0, 1))));
  gCam.lookAt(gCtl.target);
  if (t >= 1) { mode = "galaxy"; gCtl.enabled = true; $("help").style.opacity = 1; }
}

// ---------------------------------------------------------------- star system scene
const sScene = new THREE.Scene();
const sCam = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.001, 5e4);
sCam.up.set(0, 0, 1);
const sCtl = mouseControls(sCam, canvas);
sCtl.enabled = false;
sCtl.minDistance = 1.3;
sCtl.maxDistance = 400;
sScene.add(starfield(4000, 2e4, 1.3));
sScene.add(new THREE.AmbientLight(0x223344, 0.35));
let sys = null;

const starMat = (color) => new THREE.ShaderMaterial({
  uniforms: { uColor: { value: color }, uTime: { value: 0 } },
  vertexShader: `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vN; varying vec3 vP; varying vec3 vV;
    void main() {
      vN = normalize(normalMatrix * normal); vP = position;
      vec4 mv = modelViewMatrix * vec4(position, 1.0); vV = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv;
      #include <logdepthbuf_vertex>
    }`,
  fragmentShader: `
    #include <logdepthbuf_pars_fragment>
    uniform vec3 uColor; uniform float uTime;
    varying vec3 vN; varying vec3 vP; varying vec3 vV;
    float h(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
    float n3(vec3 p) {
      vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(h(i), h(i + vec3(1,0,0)), f.x), mix(h(i + vec3(0,1,0)), h(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(h(i + vec3(0,0,1)), h(i + vec3(1,0,1)), f.x), mix(h(i + vec3(0,1,1)), h(i + vec3(1,1,1)), f.x), f.y), f.z);
    }
    float fbm(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * n3(p); p *= 2.1; a *= 0.5; } return s; }
    void main() {
      float mu = clamp(dot(normalize(vN), normalize(vV)), 0.0, 1.0);
      float g = fbm(vP * 6.0 + vec3(0.0, 0.0, uTime * 0.08));
      float spots = smoothstep(0.62, 0.75, fbm(vP * 2.2 - uTime * 0.01));
      float limb = 0.35 + 0.65 * pow(mu, 0.55);
      vec3 c = uColor * limb * (0.82 + 0.38 * g) * (1.0 - 0.35 * spots);
      c += vec3(1.0, 0.95, 0.85) * pow(mu, 3.0) * 0.25;
      gl_FragColor = vec4(c, 1.0);
      #include <logdepthbuf_fragment>
    }`,
});

function planetTexture(radiusRe, seed) {
  const r = rand(seed), c = document.createElement("canvas"); c.width = 256; c.height = 128;
  const g = c.getContext("2d");
  const palette = radiusRe > 6 ? ["#c9a274", "#e3c9a0", "#a8754d", "#f0dcc0", "#8c5a3c"]
    : radiusRe > 1.8 ? ["#6fb7d9", "#8fd3ea", "#4f8fc0", "#b6e6f2", "#3f74a8"]
    : ["#8a6f5a", "#a8896b", "#5f7f9a", "#c8b08a", "#6d5444"];
  for (let y = 0; y < 128; y += 2) {
    g.fillStyle = palette[Math.floor((Math.sin(y * (radiusRe > 6 ? 0.18 : 0.07) + r() * 2) * 0.5 + 0.5) * palette.length) % palette.length];
    g.fillRect(0, y, 256, 2);
  }
  if (radiusRe <= 1.8) for (let k = 0; k < 40; k++) { g.fillStyle = `rgba(255,255,255,${0.08 + r() * 0.12})`; g.beginPath(); g.ellipse(r() * 256, r() * 128, 8 + r() * 30, 3 + r() * 8, 0, 0, 7); g.fill(); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

const markerTex = (() => {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"); g.strokeStyle = "#fff"; g.lineWidth = 6;
  g.beginPath(); g.arc(64, 64, 44, 0, Math.PI * 2); g.stroke();
  for (let k = 0; k < 4; k++) { const a = (k * Math.PI) / 2; g.beginPath(); g.moveTo(64 + 50 * Math.cos(a), 64 + 50 * Math.sin(a)); g.lineTo(64 + 62 * Math.cos(a), 64 + 62 * Math.sin(a)); g.stroke(); }
  return new THREE.CanvasTexture(c);
})();

// Orbit radius on screen: true a/R* compressed logarithmically so wide orbits still fit.
const orbitVis = (aOverR) => 1.8 + 2.6 * Math.log2(1 + aOverR / 4);

function enterSystem(i) {
  const st = DATA.positives[i];
  mode = "system";
  const srad = st.srad ?? 1;
  const color = teffColor(st.teff);
  const star = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), starMat(color));
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(`rgba(${color.r * 255},${color.g * 255},${color.b * 255},0.9)`,
    `rgba(${color.r * 255},${color.g * 255},${color.b * 255},0.18)`), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  glow.scale.set(7, 7, 1);
  const light = new THREE.PointLight(color.clone().lerp(new THREE.Color(1, 1, 1), 0.5), 2.2, 0, 0);
  const group = new THREE.Group(); group.add(star, glow, light);
  sScene.add(group);

  const planets = [];
  const shown = st.signals.filter((s) => s.positive);
  const minP = Math.min(...shown.map((s) => s.period || 10));
  shown.forEach((s, k) => {
    const rp = planetRadius(s, st);
    const aAU = s.sma ?? Math.pow((s.period || 10) / 365.25, 2 / 3);
    const aR = (aAU * 215.032) / srad;
    const orbit = orbitVis(aR) + k * 0.05;
    const size = Math.max(rp.ratio, 0.012);
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 48, 32),
      new THREE.MeshStandardMaterial({ map: planetTexture(rp.re, k * 97 + st.kepid), roughness: 0.9, metalness: 0 }));
    const hit = new THREE.Mesh(new THREE.SphereGeometry(Math.max(size * 3, 0.25), 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
    mesh.add(hit);
    const ring = new THREE.Mesh(new THREE.RingGeometry(orbit - 0.006, orbit + 0.006, 256),
      new THREE.MeshBasicMaterial({ color: 0x7fe7ff, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
    // True-size planets are often sub-pixel, so each one carries a constant-size holographic marker.
    const marker = new THREE.Sprite(new THREE.SpriteMaterial({ map: markerTex, color: 0x7fe7ff, transparent: true,
      depthWrite: false, depthTest: false, sizeAttenuation: false, blending: THREE.AdditiveBlending }));
    marker.scale.set(0.045, 0.045, 1);
    mesh.add(marker);
    const tilt = new THREE.Group(); tilt.rotation.x = (k % 2 ? -1 : 1) * 0.02 * k; tilt.add(ring, mesh);
    group.add(tilt);
    const speed = (0.55 * Math.pow(minP / (s.period || minP), 0.45));
    planets.push({ s, mesh, hit, ring, marker, orbit, speed, phase: k * 2.1 + R() * 0.5, size });
  });

  const outer = Math.max(...planets.map((p) => p.orbit), 3);
  sCam.position.set(outer * 0.45, -outer * 1.45, outer * 0.55);
  shiftForPanel();
  sCtl.target.set(0, 0, 0);
  sCtl.maxDistance = outer * 12;
  sCtl.enabled = true; gCtl.enabled = false;
  sys = { st, group, star, planets, sel: 0, galaxyPos: new THREE.Vector3(st.x, st.y, st.z) };
  $("back").hidden = false;
  showHolo(st, 0);
}
// Centre the star in the space left of the hologram panel.
function shiftForPanel() {
  const panel = Math.min(460, innerWidth - 44) + 22;
  const shift = innerWidth > 700 ? panel / 2 : 0;
  sCam.setViewOffset(innerWidth, innerHeight, shift, 0, innerWidth, innerHeight);
  sCam.updateProjectionMatrix();
}
function disposeSystem() {
  if (!sys) return;
  sScene.remove(sys.group);
  sys.group.traverse((o) => { o.geometry?.dispose(); if (o.material) { o.material.map?.dispose(); o.material.dispose(); } });
  sys = null; sCtl.enabled = false; $("back").hidden = true;
}
function planetRadius(s, st) {
  const srad = st.srad ?? 1;
  if (s.prad != null) return { re: s.prad, ratio: (s.prad * 0.0091577) / srad, est: false };
  const ratio = Math.sqrt(Math.max(s.depth_ppm ?? 100, 1) * 1e-6); // transit depth ≈ (Rp/R*)^2
  return { re: ratio * srad / 0.0091577, ratio, est: true };
}
function stepSystem(t) {
  sys.star.material.uniforms.uTime.value = t;
  sys.star.rotation.z = t * 0.03;
  sys.planets.forEach((p, k) => {
    const a = p.phase + t * p.speed;
    p.mesh.position.set(p.orbit * Math.cos(a), p.orbit * Math.sin(a), 0);
    p.mesh.rotation.z = t * 0.4;
    const on = k === sys.sel;
    p.ring.material.opacity = on ? 0.55 : 0.18;
    p.marker.material.opacity = on ? 0.95 : 0.4;
    p.marker.scale.setScalar(on ? 0.06 + 0.008 * Math.sin(t * 4) : 0.04);
  });
}
const ray = new THREE.Raycaster();
function pickPlanet(cx, cy) {
  if (!sys) return;
  ray.setFromCamera(new THREE.Vector2((cx / innerWidth) * 2 - 1, -(cy / innerHeight) * 2 + 1), sCam);
  const hits = ray.intersectObjects(sys.planets.flatMap((p) => [p.hit, p.marker]), false);
  if (hits.length) {
    const k = sys.planets.findIndex((p) => p.hit === hits[0].object || p.marker === hits[0].object);
    const sigIdx = sys.st.signals.indexOf(sys.planets[k].s);
    selectSignal(sigIdx);
  }
}

// ---------------------------------------------------------------- hologram panel
const lcCache = new Map();
async function loadLC(kepid) {
  if (!lcCache.has(kepid)) lcCache.set(kepid, fetch(`data/lc/${kepid}.json`).then((r) => r.json()));
  return lcCache.get(kepid);
}
let holoState = null;
function showHolo(st, sigIdx) {
  holoState = { st, sigIdx, view: "loc" };
  const h = $("holo"); h.hidden = false; h.classList.remove("enter"); void h.offsetWidth; h.classList.add("enter");
  renderHolo();
}
function hideHolo() { $("holo").hidden = true; holoState = null; }
function selectSignal(i) {
  if (!holoState) return;
  holoState.sigIdx = i;
  if (sys) sys.sel = Math.max(0, sys.planets.findIndex((p) => p.s === holoState.st.signals[i]));
  renderHolo();
}

function ring(p) {
  const r = 40, c = 2 * Math.PI * r, off = c * (1 - p);
  return `<svg class="prob-ring" viewBox="0 0 92 92"><circle cx="46" cy="46" r="${r}" fill="none" stroke="rgba(127,231,255,0.15)" stroke-width="6"/>
    <circle cx="46" cy="46" r="${r}" fill="none" stroke="#7fe7ff" stroke-width="6" stroke-linecap="butt" stroke-dasharray="${c}" stroke-dashoffset="${off}"
      transform="rotate(-90 46 46)" style="filter: drop-shadow(0 0 5px #3cf)"/></svg>`;
}
const LABEL_TEXT = { PC: "Kepler DR24: planet candidate", AFP: "Kepler DR24: astrophysical false positive", NTP: "Kepler DR24: not transit-like" };
const CLASS_TEXT = { planet: "Confirmed planet", candidate: "Planet candidate", false_positive: "False positive", no_planet: "No planet" };
const fmt = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

function renderHolo() {
  const { st, sigIdx, view } = holoState;
  const s = st.signals[sigIdx];
  const rp = planetRadius(s, st);
  const tabs = st.signals.length > 1 ? `<div class="h-section"><div class="h-title">SIGNALS ON THIS STAR</div><div class="tabs">${
    st.signals.map((x, i) => `<button class="tab ${i === sigIdx ? "on" : ""} ${x.positive ? "" : "neg"} ${isDisagree(x) ? "dis" : ""}" data-i="${i}">${x.id.split(".").pop()} · ${fmt(x.period, 2)} d · ${pct(x.p_planet)}</button>`).join("")}</div></div>` : "";
  const honest = s.split === "test"
    ? `<div class="note held"><span class="badge">HELD-OUT</span>The model never saw this star in training, so this is an honest score.</div>`
    : `<div class="note warn"><span class="badge">${s.split.toUpperCase()} SET</span>This star was in the model's ${s.split === "train" ? "training" : "validation"} data, so the score is optimistic.</div>`;
  const truth = [s.label && LABEL_TEXT[s.label] ? LABEL_TEXT[s.label] : s.label, s.koi_disposition ? `KOI ${s.koi}: ${s.koi_disposition.toLowerCase()}${s.name ? ` (${s.name})` : ""}` : null].filter(Boolean).join(" · ");
  $("holo-body").innerHTML = `
    <div class="h-kicker">TARGET ACQUIRED</div>
    <div class="h-name">${s.name ?? starName(st)}</div>
    <div class="h-sub">KIC ${st.kepid} · ${fmtDist(st.dist_pc)}${st.dist_err_lo != null ? ` (−${Math.round(st.dist_err_lo)}/+${Math.round(st.dist_err_hi)} pc)` : ""} · ${st.dist_source.replace("_", " ")}</div>

    <div class="h-section">
      <div class="prob">
        ${ring(s.p_planet ?? 0)}
        <div><div class="prob-num">${pct(s.p_planet)}</div><div class="prob-cap">PLANET PROBABILITY</div>
          <div class="prob-cap" style="margin-top:4px">model: ${CLASS_TEXT[s.pred].toLowerCase()}</div></div>
      </div>
      <div class="bars">${DATA.classes.map((c, k) => `<span>${CLASS_TEXT[c]}</span><span class="bar"><i style="width:${(s.p[k] * 100).toFixed(1)}%"></i></span><span class="v">${pct(s.p[k])}</span>`).join("")}</div>
      ${honest}
      ${truth ? `<div class="note">Catalog: ${truth}</div>` : ""}
      ${isDisagree(s) ? `<div class="note dis"><span class="badge">MODEL DISAGREES</span>The catalog still lists this as a candidate, but the model rates it like a confirmed planet${s.split === "train" ? " (training star, so this is weak evidence)" : ""}. Worth follow-up, not a discovery.${rp.re != null && rp.re > 20 ? " At over 20 R⊕ (about 2 Jupiters) it may be a binary star or a blend." : ""}</div>` : ""}
    </div>
    ${tabs}
    <div class="h-section">
      <div class="h-title"><span>${view === "glob" ? "FOLDED LIGHT CURVE · FULL ORBIT" : view === "loc" ? "FOLDED LIGHT CURVE · TRANSIT" : "ODD vs EVEN TRANSITS"}</span>
        <span class="tabs">
          <button class="tab ${view === "loc" ? "on" : ""}" data-v="loc">TRANSIT</button>
          <button class="tab ${view === "glob" ? "on" : ""}" data-v="glob">ORBIT</button>
          <button class="tab ${view === "oddeven" ? "on" : ""}" data-v="oddeven">ODD/EVEN</button>
        </span></div>
      <canvas class="lc" id="lc"></canvas>
      <div class="note">Every observed transit stacked at period ${fmt(s.period, 4)} d. ${view === "oddeven" ? "Different odd/even depths hint at an eclipsing binary." : ""}</div>
    </div>
    <div class="h-section facts">
      <div><div class="k">PERIOD</div><div class="v">${fmt(s.period, 3)} d</div></div>
      <div><div class="k">DURATION</div><div class="v">${fmt(s.duration_h, 2)} h</div></div>
      <div><div class="k">DEPTH</div><div class="v">${s.depth_ppm == null ? "—" : Math.round(s.depth_ppm).toLocaleString()} ppm</div></div>
      <div><div class="k">PLANET RADIUS</div><div class="v">${fmt(rp.re, 2)} R⊕</div>${rp.est ? `<div class="est">estimated from depth</div>` : ""}</div>
      <div><div class="k">STAR RADIUS</div><div class="v">${st.srad == null ? "—" : fmt(st.srad, 2) + " R☉"}</div>${st.srad == null ? `<div class="est">unknown, drawn as 1 R☉</div>` : ""}</div>
      <div><div class="k">STAR TEMP</div><div class="v">${st.teff == null ? "—" : Math.round(st.teff).toLocaleString() + " K"}</div></div>
      <div><div class="k">ORBIT</div><div class="v">${s.sma == null ? "—" : fmt(s.sma, 3) + " AU"}</div></div>
      <div><div class="k">RA / DEC</div><div class="v">${fmt(st.ra, 3)}° / ${fmt(st.dec, 3)}°</div></div>
    </div>
    <div class="note">Star and planet sizes are drawn to scale with each other; orbit distances are compressed to fit.</div>`;
  $("holo-body").querySelectorAll("[data-i]").forEach((b) => (b.onclick = () => selectSignal(+b.dataset.i)));
  $("holo-body").querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => { holoState.view = b.dataset.v; renderHolo(); }));
  loadLC(st.kepid).then((lc) => { if (holoState && holoState.st === st) drawLC($("lc"), lc[s.id], view, s); });
}

function drawLC(cv, lc, view, s) {
  if (!cv || !lc) return;
  const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const series = view === "glob" ? [{ y: lc.glob, c: "#7fe7ff" }] : view === "loc" ? [{ y: lc.loc, c: "#7fe7ff" }]
    : [{ y: lc.odd, c: "#7fe7ff", name: "odd" }, { y: lc.even, c: "#ff9de2", name: "even" }];
  const [x0, x1] = view === "glob" ? [-0.5, 0.5] : [-3, 3];
  const all = series.flatMap((q) => q.y).filter((v) => v != null);
  if (!all.length) { g.fillStyle = "#7fe7ff"; g.fillText("no data", 10, 20); return; }
  const sorted = [...all].sort((a, b) => a - b);
  let lo = sorted[Math.floor(sorted.length * 0.01)], hi = sorted[Math.floor(sorted.length * 0.99)];
  const pad = (hi - lo) * 0.15 || 1; lo -= pad; hi += pad;
  const L = 58, Rr = 8, T = 8, B = 34, pw = W - L - Rr, ph = H - T - B;
  const X = (x) => L + ((x - x0) / (x1 - x0)) * pw, Y = (y) => T + (1 - (y - lo) / (hi - lo)) * ph;
  g.font = "10px 'Share Tech Mono', monospace"; g.strokeStyle = "rgba(127,231,255,0.18)"; g.fillStyle = "rgba(127,231,255,0.65)"; g.lineWidth = 1;
  g.textAlign = "right";
  for (let k = 0; k <= 4; k++) { const y = lo + ((hi - lo) * k) / 4; g.beginPath(); g.moveTo(L, Y(y)); g.lineTo(W - Rr, Y(y)); g.stroke(); g.fillText(Math.round(y).toString(), L - 4, Y(y) + 3); }
  g.textAlign = "left";
  const ticks = view === "glob" ? [-0.5, -0.25, 0, 0.25, 0.5] : [-3, -2, -1, 0, 1, 2, 3];
  g.textAlign = "center";
  ticks.forEach((x) => { g.beginPath(); g.moveTo(X(x), T); g.lineTo(X(x), T + ph); g.stroke(); g.fillText(String(x), X(x), T + ph + 12); });
  g.fillText(view === "glob" ? "orbital phase" : "time from mid-transit (transit durations)", L + pw / 2, H - 4);
  g.save(); g.translate(12, T + ph / 2); g.rotate(-Math.PI / 2); g.fillText("Δ flux (ppm)", 0, 0); g.restore();
  g.textAlign = "left";
  for (const q of series) {
    const n = q.y.length;
    g.shadowColor = q.c; g.shadowBlur = 8; g.strokeStyle = q.c; g.fillStyle = q.c; g.lineWidth = 1.4;
    g.beginPath(); let pen = false;
    q.y.forEach((v, k) => { const x = x0 + ((k + 0.5) / n) * (x1 - x0); if (v == null) { pen = false; return; } if (!pen) g.moveTo(X(x), Y(v)); else g.lineTo(X(x), Y(v)); pen = true; });
    g.globalAlpha = 0.55; g.stroke(); g.globalAlpha = 1;
    q.y.forEach((v, k) => { if (v == null) return; const x = x0 + ((k + 0.5) / n) * (x1 - x0); g.beginPath(); g.arc(X(x), Y(v), 1.8, 0, 7); g.fill(); });
    g.shadowBlur = 0;
  }
  if (series.length > 1) series.forEach((q, k) => { g.fillStyle = q.c; g.fillText(q.name, W - Rr - 60 + k * 32, T + 12); });
}

// ---------------------------------------------------------------- search
const search = $("search"), results = $("search-results");
let hits = [], active = 0;
search.addEventListener("input", () => {
  const q = search.value.trim().toLowerCase();
  hits = [];
  if (q.length >= 2 && DATA) {
    for (let i = 0; i < DATA.positives.length && hits.length < 12; i++) {
      const st = DATA.positives[i], name = starName(st).toLowerCase();
      if (String(st.kepid).startsWith(q.replace(/^kic\s*/, "")) || name.includes(q) || st.signals.some((s) => s.koi?.toLowerCase().includes(q))) hits.push(i);
    }
  }
  active = 0; drawHits();
});
function drawHits() {
  results.innerHTML = hits.map((i, k) => { const st = DATA.positives[i]; return `<div class="${k === active ? "active" : ""}" data-i="${i}">${starName(st)}<span>${pct(st.p_planet)}</span></div>`; }).join("");
  results.querySelectorAll("div").forEach((d) => (d.onclick = () => go(+d.dataset.i)));
}
function go(i) {
  results.innerHTML = ""; search.value = ""; search.blur();
  if (mode === "system") { disposeSystem(); hideHolo(); mode = "galaxy"; gCtl.enabled = true; }
  if (mode === "galaxy") swoopTo(i);
}
search.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { active = Math.min(active + 1, hits.length - 1); drawHits(); e.preventDefault(); }
  else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); drawHits(); e.preventDefault(); }
  else if (e.key === "Enter" && hits.length) go(hits[active]);
  else if (e.key === "Escape") { results.innerHTML = ""; search.blur(); }
});
addEventListener("keydown", (e) => { if (e.key === "Escape" && document.activeElement !== search) startReturn(); });
$("back").onclick = startReturn;

// ---------------------------------------------------------------- loop
function resize() {
  renderer.setSize(innerWidth, innerHeight);
  for (const c of [gCam, sCam]) { c.aspect = innerWidth / innerHeight; c.updateProjectionMatrix(); }
  shiftForPanel();
  galaxyMat.uniforms.uScale.value = innerHeight / (2 * Math.tan((gCam.fov * Math.PI) / 360)) * PX;
  if (holoState) renderHolo();
}
addEventListener("resize", resize);

function frame() {
  const t = clock.getElapsedTime();
  posMat.uniforms.uTime.value = t;
  if (mode === "swoop") stepSwoop();
  else if (mode === "return") stepReturn();
  if (mode === "system") {
    stepSystem(t); sCtl.update(); renderer.render(sScene, sCam);
  } else {
    if (mode === "galaxy") gCtl.update();
    galaxyMat.uniforms.uScale.value = innerHeight / (2 * Math.tan((gCam.fov * Math.PI) / 360)) * PX;
    renderer.render(gScene, gCam);
  }
  updateLabels(gCam);
  requestAnimationFrame(frame);
}

async function init() {
  buildGalaxy();
  DATA = await (await fetch("data/galaxy.json")).json();
  buildKepler();
  addLabel("SUN", new THREE.Vector3(DATA.sun.x, DATA.sun.y, DATA.sun.z));
  addLabel("GALACTIC CENTRE", new THREE.Vector3(0, 0, 0), "gc");
  resize();
  frame();
  $("loading").style.opacity = 0;
  setTimeout(() => $("loading").remove(), 900);
  window.__ui = { DATA, gCam, gCtl, sCam, swoopTo, startReturn, get mode() { return mode; }, get sys() { return sys; } }; // for debugging
}
init().catch((e) => { $("loading").textContent = "FAILED TO LOAD DATA — run export_ui_data.py"; console.error(e); });
