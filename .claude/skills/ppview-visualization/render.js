#!/usr/bin/env node
/**
 * Render patchy-particle files in ppview (headless Chrome) and save a PNG.
 *
 *   node render.js --out conf.png init.top last_conf.dat input
 *
 * Starts its own static server for ppview's committed `build/` and its own
 * headless Chrome (random ports, throwaway profile), drops the files into the
 * app exactly as a user would, optionally steps to a frame, screenshots, and
 * cleans everything up. Needs no Rust toolchain and no dev server.
 *
 * Options
 *   --out FILE        PNG to write (default ppview.png)
 *   --ppview DIR      ppview checkout (default $PPVIEW_DIR, else ~/git/ppview-matthies)
 *   --frame first|last|N   trajectory frame to show (default: whatever loads first)
 *   --size WxH        viewport (default 1400x900)
 *   --settle MS       extra wait after load/frame step before the shot (default 1500)
 *   --chrome PATH     Chrome binary (default $CHROME_PATH, else google-chrome/chromium)
 *   --click LABEL     click the control with this aria-label (repeatable), e.g.
 *                     "Simulation box", "Particle legend", "Patch legend", "Bonds"
 *   --zoom N          N mouse-wheel notches over the scene; positive zooms out, negative in
 *                     (ppview frames a small system tiny - try --zoom -8)
 *   --close-panels    close floating panels (loading a cluster observable opens the
 *                     Clustering panel over the scene)
 *   --hide-controls   collapse the bottom control bar
 *
 * Everything after the options is a file to drop (any order; ppview detects
 * file types by content).
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const net = require('net');
const { spawn } = require('child_process');

function parseArgs(argv) {
  const opts = { out: 'ppview.png', size: '1400x900', settle: 1500, files: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === '--out') opts.out = val();
    else if (a === '--ppview') opts.ppview = val();
    else if (a === '--frame') opts.frame = val();
    else if (a === '--size') opts.size = val();
    else if (a === '--settle') opts.settle = Number(val());
    else if (a === '--chrome') opts.chrome = val();
    else if (a === '--click') (opts.clicks ||= []).push(val());
    else if (a === '--zoom') opts.zoom = Number(val());
    else if (a === '--close-panels') opts.closePanels = true;
    else if (a === '--hide-controls') opts.hideControls = true;
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else opts.files.push(path.resolve(a));
  }
  return opts;
}

const freePort = () => new Promise((resolve, reject) => {
  const s = net.createServer();
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
  s.on('error', reject);
});

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.wasm': 'application/wasm', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
};

// Serves buildDir under /ppview/ (the app's configured homepage path).
function serve(buildDir, port) {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (!p.startsWith('/ppview')) { res.writeHead(404); return res.end(); }
    p = p.slice('/ppview'.length) || '/';
    if (p.endsWith('/')) p += 'index.html';
    const file = path.join(buildDir, p);
    if (!file.startsWith(buildDir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); return res.end();
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(r => server.listen(port, '127.0.0.1', () => r(server)));
}

async function waitFor(check, ms, label) {
  const t0 = Date.now();
  for (;;) {
    try { if (await check()) return; } catch (_) { /* retry */ }
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${label}`);
    await new Promise(r => setTimeout(r, 200));
  }
}

(async () => {
  const opts = parseArgs(process.argv.slice(2));
  if (!opts.files.length) throw new Error('no files given');
  for (const f of opts.files) if (!fs.existsSync(f)) throw new Error(`no such file: ${f}`);

  const ppview = path.resolve(opts.ppview || process.env.PPVIEW_DIR || path.join(os.homedir(), 'git/ppview-matthies'));
  const buildDir = path.join(ppview, 'build');
  const driverPath = path.join(ppview, 'visual-tests/driver.js');
  if (!fs.existsSync(path.join(buildDir, 'index.html'))) throw new Error(`no ${buildDir}/index.html - is --ppview right?`);
  if (!fs.existsSync(path.join(ppview, 'node_modules/ws'))) {
    throw new Error(`ws not found: run "npm install --ignore-scripts" in ${ppview} (driver.js needs it)`);
  }
  // driver.js require()s 'ws', which resolves from the ppview checkout.
  const { connect, openPage } = require(driverPath);

  const chrome = opts.chrome || process.env.CHROME_PATH ||
    ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
     '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(p => fs.existsSync(p));
  if (!chrome) throw new Error('no Chrome found; pass --chrome or set CHROME_PATH');

  const [width, height] = opts.size.split('x').map(Number);
  const webPort = await freePort();
  const debugPort = await freePort();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'ppview-render-'));
  const server = await serve(buildDir, webPort);
  const chromeProc = spawn(chrome, [
    '--headless=new', '--no-sandbox', `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`, '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist', 'about:blank',
  ], { stdio: 'ignore' });
  const cleanup = () => {
    try { chromeProc.kill('SIGTERM'); } catch (_) {}
    try { server.close(); } catch (_) {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (_) {}
  };
  process.on('exit', cleanup);

  try {
    await waitFor(() => new Promise((ok, no) =>
      http.get({ host: '127.0.0.1', port: debugPort, path: '/json/version' }, r => { r.resume(); ok(true); }).on('error', no)),
      20000, 'Chrome');

    const page = await connect(debugPort, await openPage(debugPort));
    await page.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    await page.goto(`http://127.0.0.1:${webPort}/ppview/`);
    await page.dropFiles(opts.files);   // throws if the scene never draws geometry

    if (opts.frame) {
      const key = opts.frame === 'last' ? 'End' : opts.frame === 'first' ? 'Home' : null;
      const press = async (k) => {
        for (const type of ['keyDown', 'keyUp']) {
          await page.send('Input.dispatchKeyEvent', { type, key: k, code: k });
        }
      };
      if (key) await press(key);
      else {
        const n = Number(opts.frame);
        if (!Number.isInteger(n) || n < 0) throw new Error('--frame must be first, last or a 0-based integer');
        await press('Home');
        for (let i = 0; i < n; i++) await press('ArrowRight');
      }
    }
    for (const label of opts.clicks || []) {
      const ok = await page.evaluate(`(() => { const b = document.querySelector('[aria-label=' + JSON.stringify(${JSON.stringify(label)}) + ']'); if (b) b.click(); return !!b; })()`);
      if (!ok) console.error(`warning: no control with aria-label "${label}"`);
    }
    if (opts.zoom) {
      for (let i = 0; i < Math.abs(opts.zoom); i++) {
        await page.send('Input.dispatchMouseEvent', { type: 'mouseWheel', x: width / 2, y: height / 2 - 100,
          deltaX: 0, deltaY: Math.sign(opts.zoom) * 100 });
      }
    }
    // Loading a cluster observable opens the Clustering panel over the scene.
    if (opts.closePanels) {
      await page.evaluate(`document.querySelectorAll('.close-button').forEach(b => b.click())`);
    }
    await new Promise(r => setTimeout(r, 300));
    const readout = await page.evaluate(`document.querySelector('.controls-panel')?.innerText.replace(/\\s+/g,' ') ?? ''`);
    if (opts.hideControls) {
      await page.evaluate(`document.querySelector('[aria-label="Hide controls"]')?.click()`);
    }
    await new Promise(r => setTimeout(r, opts.settle));
    await page.screenshot(path.resolve(opts.out));

    const bad = page.logs.filter(l => l.level === 'exception' || l.level === 'error');
    console.log(`wrote ${path.resolve(opts.out)}`);
    console.log(`readout: ${readout}`);
    if (bad.length) console.log('page errors:', JSON.stringify(bad.slice(0, 5)));
    page.close();
  } finally {
    cleanup();
  }
  process.exit(0);
})().catch(e => { console.error(`render.js: ${e.message}`); process.exit(1); });
