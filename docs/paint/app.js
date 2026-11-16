/* The easel page: loads a stroke tape and lets atelier.js lay it, a few milliseconds per frame. */
(function () {
  'use strict';
  const PAINTINGS = [
    { id: 'harbour_at_first_light', title: 'Harbour at first light' },
    { id: 'rain_at_blue_hour', title: 'Rain at blue hour' },
    { id: 'wind_over_the_barley', title: 'Wind over the barley' },
    { id: 'lemons_and_blue_jug', title: 'Lemons and a blue jug' },
  ];
  const SECONDS = 50;                                   // one painting at 1x
  const $ = (id) => document.getElementById(id);
  const canvas = $('easel'), ctx = canvas.getContext('2d'), veil = $('veil');
  const qs = new URLSearchParams(location.search);
  const small = Math.min(window.innerWidth, window.innerHeight) < 720 || (navigator.deviceMemory && navigator.deviceMemory <= 2);
  const HEIGHT = Math.max(240, Math.min(1400, parseInt(qs.get('h'), 10) || (small ? 560 : 800)));
  const tapes = {};
  let P = null, current = null, playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let vt = 0, last = 0, seekTo = null, relight = false, token = 0, frameNo = 0, t0 = 0, spent = 0;
  window.__easel = { get p() { return P; }, get spent() { return spent; } };      // for tests

  const tabs = $('tabs');
  PAINTINGS.forEach((it) => {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.textContent = it.title; b.dataset.id = it.id;
    b.addEventListener('click', () => open(it.id));
    tabs.appendChild(b);
  });

  function say(msg) { veil.textContent = msg; veil.hidden = !msg; }

  async function open(id) {
    const mine = ++token, it = PAINTINGS.find((p) => p.id === id);
    current = id; P = null;
    for (const b of tabs.children) b.setAttribute('aria-selected', b.dataset.id === id ? 'true' : 'false');
    $('title').textContent = it.title;
    say('Fetching the strokes…');
    try {
      const tape = tapes[id] || (tapes[id] = await Atelier.loadTape('tapes/' + id + '.tape'));
      if (mine !== token) return;
      say('Stretching the canvas…');
      await new Promise((r) => setTimeout(r, 30));      // let the message show before the heavy part
      if (mine !== token) return;
      const p = new Atelier.Painting(tape, HEIGHT);
      p.light = [lightX, lightY]; p.relief = $('relief').checked ? 1 : 0;
      canvas.width = p.W; canvas.height = p.H;
      p.paint(ctx, true);
      P = p; vt = 0; seekTo = null; spent = 0; t0 = performance.now();
      $('facts').innerHTML = '<b>' + tape.strokes.toLocaleString('en') + '</b> strokes · ' + p.W + ' × ' + p.H + '<br>12 bands of pigment per pixel<br>bristles, dry brush, wet-into-wet<br>no image model, no libraries';
      say('');
      history.replaceState(null, '', '#' + id);
    } catch (e) {
      if (mine === token) say(String(e.message || e));
    }
  }

  function frame(t) {
    const dt = Math.min(0.1, (t - last) / 1000 || 0); last = t; frameNo++;
    if (P) {
      const a = performance.now();
      if (seekTo !== null) {
        if (P.advance(seekTo, 20)) { vt = P.progress * SECONDS; seekTo = null; P.paint(ctx); }
        else if (frameNo % 5 === 0) P.paint(ctx);
      } else {
        if (playing && !P.finished) {
          const sp = $('speed').value;
          if (sp === 'max') { P.advance(1, 11); vt = P.progress * SECONDS; }
          else { vt += dt * parseFloat(sp); if (!P.advance(vt / SECONDS, 9)) vt = P.progress * SECONDS; }
        }
        P.paint(ctx);
      }
      if (relight) { P.paint(ctx, 'light'); relight = false; }
      spent += performance.now() - a;
      const tape = P.tape;
      $('count').innerHTML = P.finished ? '<b>' + tape.strokes.toLocaleString('en') + '</b> strokes laid. The paint is still wet.'
        : 'stroke <b>' + P.done.toLocaleString('en') + '</b> of ' + tape.strokes.toLocaleString('en');
      if (document.activeElement !== $('seek') || seekTo === null) if (!dragging) $('seek').value = Math.round(P.progress * 1000);
      $('play').textContent = P.finished ? 'Again' : playing ? 'Pause' : 'Play';
    }
    requestAnimationFrame(frame);
  }

  function restart() { if (!P) return; P.reset(); P.paint(ctx, true); vt = 0; seekTo = null; }
  $('play').addEventListener('click', () => { if (P && P.finished) { restart(); playing = true; } else playing = !playing; });
  $('restart').addEventListener('click', () => { restart(); playing = true; });
  let dragging = false;
  const seek = $('seek');
  seek.addEventListener('pointerdown', () => { dragging = true; });
  window.addEventListener('pointerup', () => { dragging = false; });
  seek.addEventListener('change', () => {
    if (!P) return;
    const f = seek.value / 1000;
    if (f < P.progress) { P.reset(); P.paint(ctx, true); }
    seekTo = f; dragging = false;
  });
  $('save').addEventListener('click', () => {
    canvas.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = (current || 'painting') + '.png';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }, 'image/png');
  });
  $('relief').addEventListener('change', () => { if (P) { P.relief = $('relief').checked ? 1 : 0; relight = true; } });

  // the lamp: the dot is where the light comes from
  let lightX = -0.55, lightY = -0.62;
  const dial = $('dial'), knob = $('knob');
  function setLight(x, y) {
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    lightX = x; lightY = y;
    knob.style.transform = 'translate(' + (x * 27).toFixed(1) + 'px,' + (y * 27).toFixed(1) + 'px)';
    const names = [['upper left', 'above', 'upper right'], ['left', 'straight on', 'right'], ['lower left', 'below', 'lower right']];
    dial.setAttribute('aria-valuetext', names[y < -0.3 ? 0 : y > 0.3 ? 2 : 1][x < -0.3 ? 0 : x > 0.3 ? 2 : 1]);
    if (P) { P.light = [x, y]; relight = true; }
  }
  function fromPointer(e) { const r = dial.getBoundingClientRect(); setLight((e.clientX - r.left - r.width / 2) / 27, (e.clientY - r.top - r.height / 2) / 27); }
  dial.addEventListener('pointerdown', (e) => { dial.setPointerCapture(e.pointerId); fromPointer(e); });
  dial.addEventListener('pointermove', (e) => { if (dial.hasPointerCapture(e.pointerId)) fromPointer(e); });
  dial.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
    if (d) { e.preventDefault(); setLight(lightX + d[0], lightY + d[1]); }
  });
  setLight(lightX, lightY);

  const wanted = location.hash.slice(1);
  open(PAINTINGS.some((p) => p.id === wanted) ? wanted : PAINTINGS[0].id);
  requestAnimationFrame(frame);
})();
