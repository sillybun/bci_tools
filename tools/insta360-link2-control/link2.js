const { setTimeout: delay } = require('node:timers/promises');

const origin = `http://localhost:${process.env.AUTOCAMERA_PORT || 5173}`;
const operations = {
  probe: ['probe'],
  lens: ['lens'],
  stop: ['move', { pan: 0, tilt: 0 }],
};

async function request(endpoint, body) {
  const response = await fetch(`${origin}/ptz/${endpoint}`, {
    method: 'POST',
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
  });
  const result = await response.json();
  if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function integer(raw, min, max) {
  if (!/^-?\d+$/.test(raw || '') || Number(raw) < min || Number(raw) > max) {
    throw new Error(`Expected an integer from ${min} to ${max}`);
  }
  return Number(raw);
}

async function main([operation, ...args]) {
  if (Object.hasOwn(operations, operation) && !args.length) return request(...operations[operation]);
  if (operation === 'zoom' && args.length === 1) {
    const zoomSteps = integer(args[0], -100, 100);
    if (!zoomSteps) throw new Error('Zoom steps must be nonzero');
    return request('zoom', { zoomSteps });
  }
  if (operation === 'move' && args.length === 3) {
    const pan = integer(args[0], -20, 20);
    const tilt = integer(args[1], -20, 20);
    const milliseconds = integer(args[2], 1, 400);
    try {
      await request('move', { pan, tilt });
      await delay(milliseconds);
    } finally {
      await request(...operations.stop);
    }
    return { ok: true, stopped: true };
  }
  throw new Error('Usage: node link2.js probe | lens | stop | zoom STEPS | move PAN TILT MILLISECONDS');
}

main(process.argv.slice(2))
  .then(result => console.log(JSON.stringify(result)))
  .catch(error => {
    console.error(JSON.stringify({ ok: false, error: error.message, detail: error.cause?.code }));
    process.exitCode = 1;
  });
