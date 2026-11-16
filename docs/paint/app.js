/* The easel page: loads a stroke tape and lets atelier.js lay it, a few milliseconds per frame. */
(function () {
  'use strict';
  const PAINTINGS = [
    { id: 'harbour_at_first_light', title: 'Harbour at first light' },
    { id: 'rain_at_blue_hour', title: 'Rain at blue hour' },
    { id: 'wind_over_the_barley', title: 'Wind over the barley' },
    { id: 'lemons_and_blue_jug', title: 'Lemons and a blue jug' },
    { id: 'ararat_dusk_into_night', title: 'Ararat, dusk into night' },
    { id: 'blank', title: 'A blank canvas' },
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
  // a moving painting: the same strokes in the light of any moment of the film
  let light = null, fast = false, film = null;          // film: { n, frames: [ImageBitmap], next, t, dir } while the evening passes
  const lights = {};
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
      const tape = tapes[id] || (tapes[id] = id === 'blank' ? Atelier.blankTape(1.5, [0.48, 0.36, 0.26], it.title) : await Atelier.loadTape('tapes/' + id + '.tape'));
      if (mine !== token) return;
      say('Stretching the canvas…');
      await new Promise((r) => setTimeout(r, 30));      // let the message show before the heavy part
      if (mine !== token) return;
      light = tape.light ? (lights[id] || (lights[id] = await Atelier.loadLight('tapes/' + tape.light.file, tape.light))) : null;
      if (mine !== token) return;
      const p = new Atelier.Painting(tape, Math.min(HEIGHT, Math.round((light ? (small ? 720 : 1050) : (small ? 1000 : 1500)) / tape.aspect)));
      stopFilm(); fast = false;
      $('hourrow').hidden = !light; $('hour').value = 0;
      if (light) p.setMoment(light, 0);
      p.light = [lightX, lightY]; p.relief = $('relief').checked ? 1 : 0;
      canvas.width = p.W; canvas.height = p.H;
      p.paint(ctx, true);
      P = p; vt = 0; seekTo = null; spent = 0; t0 = performance.now(); mine_n = 0; $('undo').disabled = true;
      playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      $('seek').disabled = $('play').disabled = $('restart').disabled = id === 'blank';
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
      if (film) filmStep(dt);
      else if (seekTo !== null) {
        if (P.advance(seekTo, 20)) { vt = P.progress * SECONDS; seekTo = null; P.paint(ctx); }
        else if (frameNo % 5 === 0) P.paint(ctx);
      } else {
        if (playing && !P.finished) {
          const sp = $('speed').value;
          if (sp === 'max' || fast) { P.advance(1, 11); vt = P.progress * SECONDS; }
          else { vt += dt * parseFloat(sp); if (!P.advance(vt / SECONDS, 9)) vt = P.progress * SECONDS; }
        }
        P.paint(ctx);
      }
      if (relight && !film) { P.paint(ctx, 'light'); relight = false; }
      if (P.finished) fast = false;
      spent += performance.now() - a;
      const tape = P.tape;
      if (film) $('count').innerHTML = film.frames.length < film.n
        ? 'painting the light of moment <b>' + (film.frames.length + 1) + '</b> of ' + film.n + ', every stroke again'
        : 'the same <b>' + tape.strokes.toLocaleString('en') + '</b> strokes, ' + film.n + ' moments of light';
      else if (mine_n) $('count').innerHTML = '<b>' + mine_n + '</b> ' + (mine_n === 1 ? 'stroke' : 'strokes') + ' of your own on the canvas' + (tape.strokes ? ', over ' + P.done.toLocaleString('en') + ' of the painter’s' : '') + '. The paint is wet: drag through it.';
      else if (current === 'blank') $('count').innerHTML = 'A toned canvas. Pick a colour and paint.';
      else $('count').innerHTML = P.finished ? '<b>' + tape.strokes.toLocaleString('en') + '</b> strokes laid. The paint is still wet.'
        : 'stroke <b>' + P.done.toLocaleString('en') + '</b> of ' + tape.strokes.toLocaleString('en');
      if (document.activeElement !== $('seek') || seekTo === null) if (!dragging) $('seek').value = Math.round((P.progress || 0) * 1000);
      $('play').textContent = film ? 'Play' : P.finished ? 'Again' : playing ? 'Pause' : 'Play';
    }
    requestAnimationFrame(frame);
  }

  function fresh() { P.reset(); P.paint(ctx, true); vt = 0; seekTo = null; mine_n = 0; $('undo').disabled = true; }   // the canvas starts over: no stroke of the hand is left to take off
  function restart() { if (!P) return; stopFilm(); fresh(); }

  // ---- the evening: lay every stroke again for each moment, keep the pictures, then let them run
  function stopFilm() {
    if (!film) return;
    film.frames.forEach((b) => b.close && b.close());
    film = null; $('evening').textContent = 'Let the evening pass';
  }
  function startFilm() {
    if (!P || !light) return;
    film = { n: small ? 10 : 16, frames: [], pending: false, t: 0, dir: 1 };
    $('evening').textContent = 'Stop';
    P.setMoment(light, 0); fresh();
  }
  function filmStep(dt) {
    const f = film;
    if (f.frames.length < f.n) {                           // still painting the moments
      if (f.pending) return;
      if (!P.finished) { P.advance(1, 28); P.paint(ctx); return; }
      P.paint(ctx, true);
      f.pending = true;
      createImageBitmap(P.img).then((bmp) => {
        if (film !== f) { bmp.close && bmp.close(); return; }
        f.frames.push(bmp); f.pending = false;
        $('hour').value = Math.round(1000 * (f.frames.length - 1) / (f.n - 1));
        if (f.frames.length < f.n) { P.setMoment(light, f.frames.length / (f.n - 1)); P.reset(); }
      });
      return;
    }
    f.t += f.dir * dt * 3;                                  // three moments a second, there and back
    if (f.t >= f.n - 1) { f.t = f.n - 1; f.dir = -1; } else if (f.t <= 0) { f.t = 0; f.dir = 1; }
    const i = Math.min(f.n - 2, f.t | 0), k = f.t - i;
    ctx.globalAlpha = 1; ctx.drawImage(f.frames[i], 0, 0);
    ctx.globalAlpha = k; ctx.drawImage(f.frames[i + 1], 0, 0); ctx.globalAlpha = 1;
    $('hour').value = Math.round(1000 * f.t / (f.n - 1));
  }
  $('evening').addEventListener('click', () => { if (film) { const t = film.t / (film.n - 1); stopFilm(); setHour(t); } else startFilm(); });
  function setHour(t) {
    if (!P || !light) return;
    stopFilm();
    P.setMoment(light, t); fresh(); playing = true; fast = true;
  }
  $('hour').addEventListener('change', () => setHour($('hour').value / 1000));
  $('play').addEventListener('click', () => { if (film) { const t = film.t / (film.n - 1); stopFilm(); setHour(t); return; } if (P && P.finished) { restart(); playing = true; } else playing = !playing; });
  $('restart').addEventListener('click', () => { restart(); playing = true; });
  let dragging = false;
  const seek = $('seek');
  seek.addEventListener('pointerdown', () => { dragging = true; });
  window.addEventListener('pointerup', () => { dragging = false; });
  seek.addEventListener('change', () => {
    if (!P) return;
    stopFilm();
    const f = seek.value / 1000;
    if (f < P.progress) fresh();
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
  $('relief').addEventListener('change', () => { if (P) { stopFilm(); P.relief = $('relief').checked ? 1 : 0; relight = true; } });

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
    if (P) { if (film) stopFilm(); P.light = [x, y]; relight = true; }
  }
  function fromPointer(e) { const r = dial.getBoundingClientRect(); setLight((e.clientX - r.left - r.width / 2) / 27, (e.clientY - r.top - r.height / 2) / 27); }
  dial.addEventListener('pointerdown', (e) => { dial.setPointerCapture(e.pointerId); fromPointer(e); });
  dial.addEventListener('pointermove', (e) => { if (dial.hasPointerCapture(e.pointerId)) fromPointer(e); });
  dial.addEventListener('keydown', (e) => {
    const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
    if (d) { e.preventDefault(); setLight(lightX + d[0], lightY + d[1]); }
  });
  setLight(lightX, lightY);

  // ---- your turn: a loaded brush on the same wet canvas
  const TUBES = [['titanium white', '#f7f5ee'], ['naples yellow', '#f0d58c'], ['cadmium yellow', '#f6c21c'], ['yellow ochre', '#c8963c'], ['cadmium red', '#d8301f'],
    ['alizarin crimson', '#8c1c2c'], ['burnt sienna', '#8a4524'], ['burnt umber', '#4a2d1c'], ['ultramarine', '#1f2f93'], ['cerulean', '#3d8cc4'],
    ['viridian', '#1c6a56'], ['sap green', '#4f6b25'], ['ivory black', '#1c1b1a']];
  let tube = TUBES[8][1], mine_n = 0, path = null, offs = null, run = 0;
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const tubesEl = $('tubes');
  TUBES.forEach(([name, h]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'tube'; b.style.background = h; b.title = name; b.setAttribute('aria-label', name);
    b.setAttribute('aria-pressed', h === tube ? 'true' : 'false');
    b.addEventListener('click', () => { tube = h; for (const o of tubesEl.children) o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
    tubesEl.appendChild(b);
  });
  function at(e) { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height]; }
  function lay() { P.userStroke(path, { rgb: hex(tube), size: $('brush').value / 1000 * P.H, knife: $('knife').checked, offs }); }
  canvas.addEventListener('pointerdown', (e) => {
    if (!P || e.button > 0) return;
    e.preventDefault(); canvas.setPointerCapture(e.pointerId);
    stopFilm(); playing = false; seekTo = null;
    P.beginStroke(); path = [at(e)]; run = 0; offs = [Math.random(), Math.random(), Math.random(), Math.random()];
    lay(); mine_n++; $('undo').disabled = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!path || !P) return;
    const p = at(e), q = path[path.length - 1];
    if (Math.hypot(p[0] - q[0], p[1] - q[1]) < 1.5) return;
    path.push(p); run += Math.hypot(p[0] - q[0], p[1] - q[1]);
    lay();
    if (run > 3 * P.H) {                                   // a very long pull: the brush is reloaded and carries on
      P.beginStroke(); path = [q, p]; run = 0; offs = [Math.random(), Math.random(), Math.random(), Math.random()];
      lay(); mine_n++;
    }
  });
  const lift = () => { path = null; };
  canvas.addEventListener('pointerup', lift); canvas.addEventListener('pointercancel', lift);
  $('undo').addEventListener('click', () => { if (P && !path) { P.undoStroke(); mine_n = Math.max(0, mine_n - 1); $('undo').disabled = true; } });

  const wanted = location.hash.slice(1);
  open(PAINTINGS.some((p) => p.id === wanted) ? wanted : PAINTINGS[0].id);
  requestAnimationFrame(frame);
})();
