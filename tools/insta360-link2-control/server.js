const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');

const port = Number(process.env.AUTOCAMERA_PORT || 5173);
let origin;
const files = {
  '/': ['index.html', 'text/html'],
  '/styles.css': ['styles.css', 'text/css'],
  '/app.js': ['app.js', 'text/javascript'],
};
let worker;
let pending;
let stopTimer;
let lastCommand = Promise.resolve();

function command(payload) {
  const request = lastCommand.then(() => runCommand(payload));
  lastCommand = request.catch(() => {});
  return request;
}

function runCommand(payload) {
  if (!worker) {
    const child = spawn('powershell.exe', [
      '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(__dirname, 'ptz.ps1'),
    ], { windowsHide: true });
    worker = child;
    createInterface({ input: child.stdout }).on('line', line => {
      if (worker !== child || !pending) return;
      try { pending(JSON.parse(line)); }
      catch { pending({ ok: false, error: '无法读取 Windows 控制结果' }); }
    });
    child.stderr.pipe(process.stderr);
    const closed = error => {
      if (worker !== child) return;
      worker = null;
      pending?.({ ok: false, error: error?.message || 'Windows 控制进程已退出' });
    };
    child.on('error', closed);
    child.on('exit', () => closed());
    child.stdin.on('error', closed);
  }
  return new Promise(resolve => {
    pending = result => {
      pending = null;
      resolve(result);
    };
    worker.stdin.write(JSON.stringify(payload) + '\n');
  });
}

function scheduleStop() {
  clearTimeout(stopTimer);
  stopTimer = setTimeout(async function stop() {
    if (pending) {
      stopTimer = setTimeout(stop, 100);
      return;
    }
    await command({ pan: 0, tilt: 0 });
  }, 420);
}

const server = http.createServer(async (req, res) => {
  if (req.headers.host !== new URL(origin).host) {
    res.writeHead(403).end();
    return;
  }
  if (req.method === 'GET' && Object.hasOwn(files, req.url)) {
    const [file, type] = files[req.url];
    res.writeHead(200, { 'Content-Type': type + '; charset=utf-8', 'Cache-Control': 'no-store' });
    fs.createReadStream(path.join(__dirname, file)).pipe(res);
    return;
  }
  const probe = req.url === '/ptz/probe';
  const lensProbe = req.url === '/ptz/lens';
  const zoom = req.url === '/ptz/zoom';
  if (req.method !== 'POST' || (!probe && !lensProbe && !zoom && req.url !== '/ptz/move')) {
    res.writeHead(404).end();
    return;
  }
  if (req.headers.origin !== origin) {
    res.writeHead(403).end();
    return;
  }
  let result;
  try {
    if (probe) {
      result = await command({ probe: true });
    } else if (lensProbe) {
      result = await command({ lensProbe: true });
    } else {
      let body = '';
      for await (const chunk of req) {
        body += chunk;
        if (body.length > 128) throw new Error('控制参数过长');
      }
      const { pan, tilt, zoomSteps } = JSON.parse(body);
      if (zoom) {
        if (!Number.isInteger(zoomSteps) || !zoomSteps || Math.abs(zoomSteps) > 100) {
          throw new Error('变焦步数超出范围');
        }
        result = await command({ zoomSteps });
      } else {
        if (![pan, tilt].every(value => Number.isInteger(value) && Math.abs(value) <= 20)) {
          throw new Error('云台速度超出范围');
        }
        result = await command({ pan, tilt });
        if (result.ok && !pan && !tilt) clearTimeout(stopTimer);
        else if (pan || tilt) scheduleStop();
      }
    }
  } catch (error) {
    result = { ok: false, error: error.message };
  }
  res.writeHead(result.ok ? 200 : 503, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(result));
});

server.listen(port, '::1', () => {
  origin = `http://localhost:${server.address().port}`;
  console.log(origin);
  if (process.argv.includes('--open') && process.platform === 'win32') {
    const browser = spawn('explorer.exe', [origin], { windowsHide: true });
    browser.on('error', error => console.error(error.message));
  }
});
