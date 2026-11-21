#!/usr/bin/env node
/**
 * Render the polycubes web app (sulcgroup/polycubes-clone, `web/`) headlessly and save a PNG.
 *
 *   gallery  (default)  structure JSON files side by side:
 *       node render.js --out s.png duck.json other.json
 *   viewer   a rule string (the same string a PolycubesRule uses) run in the browser's own
 *            stochastic assembler, or an uploaded structure JSON:
 *       node render.js --page viewer --rule "1|1|1|1|1|1_-1|||||" --steps 60 --out v.png
 *       node render.js --page viewer --structure duck.json --out v.png
 *   solve    a SAT solve spec JSON loaded into the solver page (does not run the solver):
 *       node render.js --page solve --spec topology.json --out sp.png
 *
 * Builds the three pages with Parcel into a cache directory the first time (and again whenever
 * the checkout's HEAD or its uncommitted web/src changes), serves them on a random port, starts a
 * throwaway headless Chrome, drives the page over the DevTools protocol, and cleans up. The
 * repo's own build directories and package.json are not touched.
 *
 * Options
 *   --out FILE        PNG to write (default polycubes.png)
 *   --repo DIR        polycubes-clone checkout (default $POLYCUBES_DIR, else ~/git/polycubes-clone)
 *   --page NAME       gallery | viewer | solve   (default gallery)
 *   --rule STR        viewer: rule string, passed as ?rule=
 *   --structure FILE  viewer: upload a structure JSON ({cube_types, cubes})
 *   --spec FILE       solve: upload a solve spec JSON
 *   --show-env        viewer: open the environment panel (the #envToggle button) in the shot
 *   --env-json FILE   viewer: write the environment panel's per-cube tables (#env-browser) as JSON:
 *                     [{name, type, position, patches: [{face, ori, color, stateVar, actVar}]}]
 *   --show-rule       viewer: make --out a PNG of the flattened cube-type diagram (what the
 *                     toggle-rule button opens) instead of the 3D scene
 *   --rule-svg FILE   viewer: also/instead save that diagram as a standalone SVG
 *   --rule-columns N  re-flow the diagram to N cube types per row (default: the app's own wrapping)
 *   --rule-scale S    scale factor for the diagram's width/height (viewBox stays put)
 *   --steps N         viewer: run the simulation until the step counter reaches N, then pause
 *   --max-cubes N     viewer: ?nMaxCubes= (default the app's 100)
 *   --seed S          viewer: ?randseed= for a repeatable run
 *   --size WxH        viewport (default 1400x900)
 *   --settle MS       wait before the shot (default 1500)
 *   --rebuild         force a rebuild of the cached pages
 *   --chrome PATH     Chrome binary (default $CHROME_PATH, else google-chrome/chromium)
 *
 * Any bare arguments are files for the page: structure JSONs for the gallery.
 * Needs `ws` resolvable from <repo>/web/node_modules: run `npm install --no-save ws` there
 * (it is already gitignored and --no-save leaves package.json alone).
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const net = require('net');
const crypto = require('crypto');
const { spawn, spawnSync } = require('child_process');

function parseArgs(argv) {
  const o = { out: 'polycubes.png', page: 'gallery', size: '1400x900', settle: 1500, files: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === '--out') o.out = val();
    else if (a === '--repo') o.repo = val();
    else if (a === '--page') o.page = val();
    else if (a === '--rule') o.rule = val();
    else if (a === '--structure') o.structure = path.resolve(val());
    else if (a === '--spec') o.spec = path.resolve(val());
    else if (a === '--steps') o.steps = Number(val());
    else if (a === '--show-rule') o.showRule = true;
    else if (a === '--show-env') o.showEnv = true;
    else if (a === '--env-json') o.envJson = path.resolve(val());
    else if (a === '--rule-svg') o.ruleSvg = path.resolve(val());
    else if (a === '--rule-columns') o.ruleColumns = Number(val());
    else if (a === '--rule-scale') o.ruleScale = Number(val());
    else if (a === '--max-cubes') o.maxCubes = Number(val());
    else if (a === '--seed') o.seed = val();
    else if (a === '--size') o.size = val();
    else if (a === '--settle') o.settle = Number(val());
    else if (a === '--rebuild') o.rebuild = true;
    else if (a === '--chrome') o.chrome = val();
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.files.push(path.resolve(a));
  }
  return o;
}

const freePort = () => new Promise((resolve, reject) => {
  const s = net.createServer();
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
  s.on('error', reject);
});
const sleep = ms => new Promise(r => setTimeout(r, ms));

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, encoding: 'utf8', maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')} failed:\n${(r.stdout || '') + (r.stderr || '')}`.slice(-3000));
  return r.stdout;
}

// Parcel builds each page separately; scope hoisting breaks the solve page (see web/package.json).
function ensureBuilt(repo, force) {
  const web = path.join(repo, 'web');
  let state = '';
  try {
    state = run('git', ['rev-parse', 'HEAD'], repo) + run('git', ['diff', 'HEAD', '--', 'web/src', 'web/package.json'], repo);
  } catch (_) { state = String(fs.statSync(path.join(web, 'src')).mtimeMs); }
  const key = crypto.createHash('sha1').update(repo + state).digest('hex').slice(0, 12);
  const cacheRoot = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  const dist = path.join(cacheRoot, 'pypatchy-polycubes-web', key);
  if (!force && fs.existsSync(path.join(dist, '.done'))) return dist;
  console.error(`building polycubes web app into ${dist} (first run for this checkout state)...`);
  fs.rmSync(dist, { recursive: true, force: true });
  fs.mkdirSync(dist, { recursive: true });
  run('npx', ['tsc'], web);
  for (const [page, extra] of [['gallery', []], ['index', []], ['solve', ['--no-scope-hoist']]]) {
    run('npx', ['parcel', 'build', `src/${page}.html`, ...extra, '--no-cache', '--dist-dir', dist, '--public-url', './'], web);
  }
  fs.writeFileSync(path.join(dist, '.done'), '');
  return dist;
}

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.map': 'application/json',
};
function serve(dir, port) {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (p === '/') p = '/index.html';
    const file = path.join(dir, p);
    if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); return res.end();
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(r => server.listen(port, '127.0.0.1', () => r(server)));
}

// Minimal Chrome DevTools Protocol client.
async function cdpConnect(WebSocket, port) {
  const targets = await new Promise((ok, no) => http.get({ host: '127.0.0.1', port, path: '/json/list' }, r => {
    let b = ''; r.on('data', c => (b += c)); r.on('end', () => ok(JSON.parse(b)));
  }).on('error', no));
  const page = targets.find(t => t.type === 'page');
  if (!page) throw new Error('no page target');
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
  let id = 0;
  const pending = new Map();
  const logs = [];
  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id); pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    } else if (m.method === 'Runtime.exceptionThrown') {
      logs.push({ level: 'exception', text: m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text });
    } else if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) {
      logs.push({ level: m.params.type, text: (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ') });
    }
  });
  await new Promise(r => ws.on('open', r));
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const n = ++id;
    const t = setTimeout(() => { pending.delete(n); reject(new Error(`${method} timed out`)); }, 60000);
    pending.set(n, { resolve: v => { clearTimeout(t); resolve(v); }, reject: e => { clearTimeout(t); reject(e); } });
    ws.send(JSON.stringify({ id: n, method, params }));
  });
  await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable');
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  return {
    send, evaluate, logs, close: () => ws.close(),
    async setFiles(selector, files) {
      const doc = await send('DOM.getDocument');
      const node = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector });
      if (!node.nodeId) throw new Error(`no element ${selector}`);
      await send('DOM.setFileInputFiles', { nodeId: node.nodeId, files });
    },
    async screenshot(file, clip) {
      const s = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
      fs.writeFileSync(file, Buffer.from(s.data, 'base64'));
    },
  };
}

(async () => {
  const o = parseArgs(process.argv.slice(2));
  const repo = path.resolve(o.repo || process.env.POLYCUBES_DIR || path.join(os.homedir(), 'git/polycubes-clone'));
  const web = path.join(repo, 'web');
  if (!fs.existsSync(path.join(web, 'package.json'))) throw new Error(`${web}/package.json not found - is --repo right?`);
  let WebSocket;
  try { WebSocket = require(require.resolve('ws', { paths: [web] })); } catch (_) {
    throw new Error(`ws not found: run "npm install --no-save ws" in ${web}`);
  }
  if (!['gallery', 'viewer', 'solve'].includes(o.page)) throw new Error('--page must be gallery, viewer or solve');
  const files = o.page === 'gallery' ? o.files : [];
  if (o.page === 'gallery' && !files.length) throw new Error('gallery needs one or more structure JSON files');
  if (o.page === 'solve' && !o.spec && o.files.length) o.spec = o.files[0];
  if (o.page === 'viewer' && !o.structure && o.files.length) o.structure = o.files[0];
  for (const f of [...files, o.structure, o.spec].filter(Boolean)) if (!fs.existsSync(f)) throw new Error(`no such file: ${f}`);

  const dist = ensureBuilt(repo, o.rebuild);
  const chrome = o.chrome || process.env.CHROME_PATH ||
    ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
     '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(p => fs.existsSync(p));
  if (!chrome) throw new Error('no Chrome found; pass --chrome or set CHROME_PATH');

  const [width, height] = o.size.split('x').map(Number);
  const webPort = await freePort();
  const debugPort = await freePort();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'polycubes-render-'));
  const server = await serve(dist, webPort);
  const chromeProc = spawn(chrome, [
    '--headless=new', '--no-sandbox', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', 'about:blank',
  ], { stdio: 'ignore' });
  const cleanup = () => {
    try { chromeProc.kill('SIGTERM'); } catch (_) {}
    try { server.close(); } catch (_) {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (_) {}
  };
  process.on('exit', cleanup);

  try {
    for (let i = 0; ; i++) {
      try { await new Promise((ok, no) => http.get({ host: '127.0.0.1', port: debugPort, path: '/json/version' }, r => { r.resume(); ok(); }).on('error', no)); break; }
      catch (e) { if (i > 100) throw new Error('Chrome did not start'); await sleep(200); }
    }
    const page = await cdpConnect(WebSocket, debugPort);
    await page.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });

    let ruleClip;
    const html = { gallery: 'gallery.html', viewer: 'index.html', solve: 'solve.html' }[o.page];
    const q = new URLSearchParams();
    if (o.page === 'viewer') {
      if (o.rule) q.set('rule', o.rule);
      if (o.maxCubes) q.set('nMaxCubes', String(o.maxCubes));
      if (o.seed) q.set('randseed', o.seed);
    }
    await page.send('Page.navigate', { url: `http://127.0.0.1:${webPort}/${html}${q.toString() ? '?' + q : ''}` });
    await sleep(1500);

    if (o.page === 'gallery') {
      await page.setFiles('#uploadfiles', files);
      await sleep(500);
      const cells = await page.evaluate(`document.querySelectorAll('.gallery-viewport').length`);
      const failed = await page.evaluate(`[...document.querySelectorAll('.gallery-label')].map(l => l.innerText).filter(t => t.includes('failed'))`);
      console.log(`gallery cells: ${cells}`);
      if (failed.length) console.log('failed to load:', JSON.stringify(failed));
    } else if (o.page === 'viewer') {
      if (o.structure) await page.setFiles('#uploadsys', [o.structure]);
      if (o.steps) {
        await page.evaluate(`document.getElementById('fff').click()`);
        // Stop at the target, or once the counter has been still for 3 s: a finite rule runs out of
        // moves (the assembly is complete) and the counter simply stops.
        const t0 = Date.now();
        let last = -1, lastChange = Date.now();
        for (;;) {
          const n = await page.evaluate(`parseInt((document.getElementById('stepcount').innerText.match(/\\d+/) || ['0'])[0])`);
          if (n >= o.steps) break;
          if (n !== last) { last = n; lastChange = Date.now(); }
          if (Date.now() - lastChange > 3000) { console.error(`note: simulation stopped advancing at step ${n} (< ${o.steps}); the assembly has likely finished`); break; }
          if (Date.now() - t0 > 120000) { console.error(`warning: gave up at step ${n} < ${o.steps}`); break; }
          await sleep(200);
        }
        await page.evaluate(`document.getElementById('pause').click()`);
      }
      console.log('readout:', await page.evaluate(`document.getElementById('stepcount')?.innerText ?? ''`));
      if (o.envJson) {
        // One <table class="env-vis-box"> per cube (present in the DOM even while the panel is hidden)
        const env = await page.evaluate(`[...document.querySelectorAll('#env-browser table.env-vis-box')].map(t => {
          const rows = [...t.querySelectorAll('tr')];
          const header = k => { const r = rows.find(r => r.querySelector('td b')?.innerText.startsWith(k)); return r ? r.querySelectorAll('td')[1].innerText.trim() : null; };
          const heads = [...(rows.find(r => r.querySelector('td[style*="border-bottom"]')) || document.createElement('tr')).querySelectorAll('td')].map(td => td.innerText.trim());
          return {
            name: t.getAttribute('name'), type: header('Cube Type'), cubeName: header('Cube Name'), position: header('Position'),
            patches: rows.filter(r => r.classList.contains('face-label')).map(r => {
              const tds = [...r.querySelectorAll('td')];
              const img = r.querySelector('img.face-img');
              const rot = img ? [...img.classList].find(c => /^rot/.test(c)) : null;
              const cells = tds.slice(1).filter(td => !td.querySelector('img')).map(td => td.innerText.trim());
              return { face: r.getAttribute('name'), ori: rot ? Number(rot.slice(3)) : null, color: Number(cells[0]),
                       stateVar: cells.length > 1 ? Number(cells[1]) : null, actVar: cells.length > 2 ? Number(cells[2]) : null };
            }),
          };
        })`);
        fs.writeFileSync(path.resolve(o.envJson), JSON.stringify(env, null, 1) + '\n');
        console.log(`environment panel: ${env.length} cubes -> ${path.resolve(o.envJson)}`);
      }
      if (o.showEnv) { await page.evaluate(`document.getElementById('envToggle').click()`); await sleep(300); }
      if (o.showRule || o.ruleSvg) {
        // Open the flattened cube-type diagram (this is what makes the app draw it), then take the
        // <svg id="rule-display-canvas"> out of the page: one <g> per cube type (a cross-net of six
        // 64px faces with patch glyphs, positioned by a translate and wrapped into rows).
        await page.evaluate(`document.getElementById('toggle-rule').click()`);
        await sleep(500);
        const ruleSvg = await page.evaluate(`((columns, scale) => {
          const svg = document.getElementById('rule-display-canvas');
          const groups = [...svg.children].filter(e => e.tagName.toLowerCase() === 'g');
          if (columns > 0) {
            // re-flow: N cube types per row, cells sized to the widest/tallest type
            const boxes = groups.map(g => g.getBBox());
            const cellW = Math.max(...boxes.map(b => b.x + b.width)) + 16;
            const cellH = Math.max(...boxes.map(b => b.y + b.height)) + 16;
            groups.forEach((g, i) => g.setAttribute('transform', 'translate(' + (i % columns) * cellW + ',' + Math.floor(i / columns) * cellH + ')'));
          }
          const b = svg.getBBox();
          const pad = 10;
          const w = Math.ceil(b.x + b.width + pad), h = Math.ceil(b.y + b.height + pad);
          svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
          svg.setAttribute('width', Math.ceil(w * scale));
          svg.setAttribute('height', Math.ceil(h * scale));
          svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          return { markup: new XMLSerializer().serializeToString(svg), width: Math.ceil(w * scale), height: Math.ceil(h * scale), types: groups.length };
        })(${o.ruleColumns || 0}, ${o.ruleScale || 1})`);
        console.log(`rule diagram: ${ruleSvg.types} cube types, ${ruleSvg.width}x${ruleSvg.height}px`);
        if (o.ruleSvg) {
          fs.writeFileSync(path.resolve(o.ruleSvg), '<?xml version="1.0" encoding="UTF-8"?>\n' + ruleSvg.markup + '\n');
          console.log(`wrote ${path.resolve(o.ruleSvg)}`);
        }
        if (!o.showRule) {
          // only the SVG was wanted: close the panel again so it doesn't cover the 3D scene
          await page.evaluate(`document.getElementById('toggle-rule').click()`);
          await sleep(200);
        }
        if (o.showRule) {
          // rasterize just the diagram: swap the page for the standalone SVG on white
          await page.send('Emulation.setDeviceMetricsOverride', { width: Math.max(300, ruleSvg.width), height: Math.max(200, ruleSvg.height), deviceScaleFactor: 1, mobile: false });
          await page.evaluate(`(markup => { document.body.style.cssText = 'margin:0;background:#fff';
            document.body.innerHTML = markup; })(${JSON.stringify(ruleSvg.markup)})`);
          ruleClip = { x: 0, y: 0, width: ruleSvg.width, height: ruleSvg.height };
        }
      }
    } else {
      if (o.spec) {
        await page.setFiles('#upload-spec', [o.spec]);
        await sleep(500);
        const err = await page.evaluate(`(() => { const e = document.getElementById('loadError'); return e && !e.hidden ? e.innerText : ''; })()`);
        if (err) console.error(`solve page rejected the spec: ${err}`);
        console.log('voxels:', await page.evaluate(`document.getElementById('voxelcount')?.innerText ?? ''`));
      }
    }
    await sleep(o.settle);
    await page.screenshot(path.resolve(o.out), ruleClip);
    console.log(`wrote ${path.resolve(o.out)}`);
    if (page.logs.length) console.log('page errors:', JSON.stringify(page.logs.slice(0, 5)));
    page.close();
  } finally {
    cleanup();
  }
  process.exit(0);
})().catch(e => { console.error(`render.js: ${e.message}`); process.exit(1); });
