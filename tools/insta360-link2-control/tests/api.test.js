const { once } = require('node:events');
const { spawn } = require('node:child_process');
const path = require('node:path');
const { test } = require('node:test');
const assert = require('node:assert/strict');

test('local API validates requests without touching the camera', { timeout: 10000 }, async () => {
  const server = spawn(process.execPath, ['server.js'], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, AUTOCAMERA_PORT: '0' },
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  try {
    const [output] = await Promise.race([
      once(server.stdout, 'data'),
      once(server, 'exit').then(code => { throw new Error(`Server exited: ${code}`); }),
    ]);
    const origin = output.toString().trim();
    const post = (route, body, requestOrigin = origin) => fetch(origin + route, {
      method: 'POST',
      headers: { Origin: requestOrigin, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(3000),
    });
    assert.equal((await fetch(origin)).status, 200);
    assert.equal((await post('/ptz/lens', null, 'https://example.com')).status, 403);
    assert.equal((await post('/ptz/unknown')).status, 404);
    for (const [route, body] of [
      ['/ptz/zoom', { zoomSteps: 0 }],
      ['/ptz/zoom', { zoomSteps: 101 }],
      ['/ptz/move', { pan: -21, tilt: 0 }],
    ]) {
      const response = await post(route, body);
      assert.equal(response.status, 503);
      assert.equal((await response.json()).ok, false);
    }
  } finally {
    if (server.kill()) await once(server, 'exit');
  }
});
