const $ = (id) => document.getElementById(id);
let stream = null;
let recorder = null;
let ptzReady = false;
const preview = $('preview');
const videoWrap = preview.parentElement;
let zoomControl = null;
const isLink2 = label => /Insta360\s+Link\s*2\b/i.test(label);
let connectionAttempt = null;

function setConnection(state, message) {
  const connected = state === 'connected';
  const connecting = state === 'connecting';
  const failed = state === 'failed';
  $('connectionDot').classList.toggle('live', connected);
  $('connectionText').textContent = connected ? '已连接' : connecting ? '连接中' : failed ? '连接失败' : '未连接';
  $('connectBtn').textContent = connected ? '断开设备' : connecting ? '取消连接' : '连接设备';
  $('recordBtn').disabled = !connected;
  $('streamBadge').textContent = connected ? 'UVC 视频流' : connecting ? '正在打开摄像头' : '等待视频源';
  $('uvcState').textContent = connected ? '已连接' : connecting ? '连接中' : failed ? '不可用' : '未探测';
  $('uvcState').classList.toggle('ok', connected);
  $('connectionMessage').textContent = message || '点击“连接设备”打开 Link 2';
  $('videoEmpty').style.display = connected ? 'none' : 'grid';
}

function stopStream(media) {
  media?.getTracks().forEach(track => track.stop());
}

function connectionError(error) {
  switch (error.name) {
    case 'NotAllowedError': return '浏览器拒绝了摄像头访问，请在当前站点允许摄像头后重试。';
    case 'NotFoundError': return '浏览器未发现 Insta360 Link 2。请检查 USB 连接。';
    case 'NotReadableError': return '无法读取 Link 2。请关闭正在使用它的应用或标签页后重试。';
    default: return error.message || '无法打开摄像头。';
  }
}

async function connect() {
  if (stream || connectionAttempt) return;
  const attempt = new AbortController();
  const { signal } = attempt;
  connectionAttempt = attempt;
  let candidate;
  const timeout = setTimeout(() => {
    if (connectionAttempt !== attempt) return;
    disconnect();
    setConnection('failed', '连接超时。请检查浏览器的摄像头权限提示后重试。');
  }, 20000);
  signal.addEventListener('abort', () => {
    clearTimeout(timeout);
    stopStream(candidate);
  }, { once: true });
  setConnection('connecting', '正在连接 Insta360 Link 2。');

  try {
    const media = navigator.mediaDevices;
    if (!media?.getUserMedia) throw new Error('请从本地服务地址打开控制台，以访问摄像头。');
    const findLink = async () => (await media.enumerateDevices())
      .find(device => device.kind === 'videoinput' && isLink2(device.label));

    let camera = await findLink();
    signal.throwIfAborted();
    if (!camera) {
      $('connectionMessage').textContent = '正在等待浏览器摄像头权限。';
      candidate = await media.getUserMedia({ video: true, audio: false });
      signal.throwIfAborted();
      if (!isLink2(candidate.getVideoTracks()[0]?.label)) {
        camera = await findLink();
        signal.throwIfAborted();
        if (!camera) throw new Error('浏览器未发现 Insta360 Link 2。请检查 USB 连接。');
        stopStream(candidate);
        candidate = null;
      }
    }

    if (!candidate) {
      const deviceId = { exact: camera.deviceId };
      $('connectionMessage').textContent = '正在打开 Insta360 Link 2 视频。';
      try {
        candidate = await media.getUserMedia({
          video: { deviceId, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 } },
          audio: false,
        });
      } catch (error) {
        signal.throwIfAborted();
        if (error.name !== 'NotReadableError' && error.name !== 'OverconstrainedError') throw error;
        candidate = await media.getUserMedia({ video: { deviceId }, audio: false });
      }
    }
    signal.throwIfAborted();

    preview.srcObject = candidate;
    await preview.play();
    signal.throwIfAborted();
    const track = candidate.getVideoTracks()[0];
    if (!track || track.readyState !== 'live') throw new Error('摄像头在启动时已断开。');
    const settings = track.getSettings();
    stream = candidate;
    $('sourceName').textContent = track.label || 'Insta360 Link 2';
    $('resolution').textContent = `${settings.width || '--'} × ${settings.height || '--'}`;
    $('fps').textContent = `${settings.frameRate || '--'} FPS`;
    $('latency').textContent = '实时';
    track.addEventListener('ended', () => {
      if (stream !== candidate) return;
      disconnect();
      setConnection('failed', '摄像头已断开。请检查 USB 连接后重试。');
    }, { once: true });
    setConnection('connected');
    void detectControlInterface();
  } catch (error) {
    stopStream(candidate);
    if (signal.aborted) return;
    preview.srcObject = null;
    setConnection('failed', connectionError(error));
  } finally {
    clearTimeout(timeout);
    if (connectionAttempt === attempt) connectionAttempt = null;
  }
}

function disconnect() {
  connectionAttempt?.abort();
  connectionAttempt = null;
  if (recorder) stopRecording();
  stopStream(stream);
  stream = null;
  preview.srcObject = null;
  videoWrap.title = '';
  zoomControl = null;
  releaseStick();
  ptzReady = false;
  $('sourceName').textContent = '未选择';
  $('resolution').textContent = '-- × --';
  $('fps').textContent = '-- FPS';
  $('latency').textContent = '-- ms';
  $('controlState').textContent = '未探测';
  $('controlState').classList.remove('ok');
  $('notice').textContent = '连接设备后显示能力检测结果';
  setConnection('idle');
}
async function ptzRequest(endpoint, values) {
  const response = await fetch('/ptz/' + endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(values),
  });
  const result = await response.json();
  if (!result.ok) throw new Error(result.error);
  return result;
}

function showZoom(range) {
  $('streamBadge').textContent = range ? `UVC · ${(range.value / 100).toFixed(2)}×` : 'UVC 视频流';
  videoWrap.title = range ? '滚轮向上放大，向下缩小' : '';
}

async function detectLens(detectedStream) {
  try {
    const { zoom } = await ptzRequest('lens');
    if (stream !== detectedStream) return;
    if (zoomControl?.stream === detectedStream && zoomControl.sending) return;
    zoomControl = zoom ? {
      stream: detectedStream,
      increment: Math.max(1, Math.round((zoom.max - zoom.min) / zoom.step / 30)),
      pending: 0,
      sending: false,
    } : null;
    showZoom(zoom);
  } catch (error) {
    if (stream === detectedStream) $('notice').textContent = `镜头探测失败：${error.message}`;
  }
}

videoWrap.addEventListener('wheel', async event => {
  const control = zoomControl;
  if (!control || control.stream !== stream || event.ctrlKey || !event.deltaY) return;
  event.preventDefault();
  const change = -Math.sign(event.deltaY) * control.increment;
  control.pending = Math.max(-100, Math.min(100, control.pending + change));
  if (control.sending) return;
  control.sending = true;
  try {
    while (zoomControl === control && control.pending) {
      const zoomSteps = control.pending;
      control.pending = 0;
      const { zoom } = await ptzRequest('zoom', { zoomSteps });
      if (zoomControl === control) showZoom(zoom);
    }
  } catch (error) {
    control.pending = 0;
    if (zoomControl === control) $('notice').textContent = `变焦失败：${error.message}`;
  } finally {
    control.sending = false;
  }
}, { passive: false });

async function detectControlInterface() {
  const detectedStream = stream;
  ptzReady = false;
  $('controlState').classList.remove('ok');
  try {
    await ptzRequest('probe');
    if (stream !== detectedStream) return;
    ptzReady = true;
    $('controlState').textContent = '云台可测试';
    $('controlState').classList.add('ok');
    $('notice').textContent = 'Windows 已读取云台控制属性';
  } catch (error) {
    if (stream !== detectedStream) return;
    $('controlState').textContent = '云台不可用';
    $('notice').textContent = error.message;
  }
  if (stream === detectedStream) await detectLens(detectedStream);
}
function setRecording(recording) {
  $('recordBtn').classList.toggle('active', recording);
  $('recordLabel').textContent = recording ? '停止录制' : '开始录制';
  $('recordState').textContent = recording ? '录制中' : '待机';
}

function stopRecording() {
  const current = recorder;
  recorder = null;
  if (current.state !== 'inactive') current.stop();
  setRecording(false);
}

function toggleRecord() {
  if (recorder) {
    stopRecording();
    return;
  }
  if (!stream) return;
  try {
    const current = new MediaRecorder(stream);
    const chunks = [];
    current.ondataavailable = event => {
      if (event.data.size) chunks.push(event.data);
    };
    current.onstop = () => {
      if (recorder === current) {
        recorder = null;
        setRecording(false);
      }
      if (!chunks.length) return;
      const file = new Blob(chunks, { type: current.mimeType || 'video/webm' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(file);
      link.href = url;
      link.download = `link2-${Date.now()}.${file.type.startsWith('video/mp4') ? 'mp4' : 'webm'}`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    current.start();
    recorder = current;
    setRecording(true);
  } catch (error) {
    $('recordState').textContent = '启动失败';
    $('notice').textContent = error.message;
  }
}
const stick = $('stick');
let pointer = null;
let pan = 0;
let tilt = 0;
let sending = false;
let stopPending = false;

async function sendPtz() {
  if (sending || (!stopPending && (!ptzReady || pointer === null))) return;
  const stop = stopPending;
  stopPending = false;
  sending = true;
  try {
    await ptzRequest('move', stop ? { pan: 0, tilt: 0 } : { pan, tilt });
    if (ptzReady) $('controlState').textContent = stop ? '停止指令已发送' : '移动指令已发送';
  } catch (error) {
    $('controlState').textContent = '发送失败';
    $('notice').textContent = error.message;
  } finally {
    sending = false;
    if (stopPending) sendPtz();
  }
}
function releaseStick() {
  if (pointer === null) return;
  pointer = null;
  pan = tilt = 0;
  stick.style.transform = 'translate(-50%,-50%)';
  $('panValue').textContent = '0';
  $('tiltValue').textContent = '0';
  stopPending = true;
  sendPtz();
}
stick.addEventListener('pointerdown', event => {
  if (!ptzReady || pointer !== null) return;
  pointer = event.pointerId;
  stick.setPointerCapture(pointer);
});
stick.addEventListener('pointermove', event => {
  if (event.pointerId !== pointer) return;
  const rect = stick.parentElement.getBoundingClientRect();
  const clamp = value => Math.max(-52, Math.min(52, value));
  const x = clamp(event.clientX - rect.left - rect.width / 2);
  const y = clamp(event.clientY - rect.top - rect.height / 2);
  pan = Math.round(x / 52 * 20);
  tilt = Math.round(y / 52 * 20);
  stick.style.transform = `translate(calc(-50% + ${x}px),calc(-50% + ${y}px))`;
  $('panValue').textContent = pan;
  $('tiltValue').textContent = tilt;
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  stick.addEventListener(event, e => { if (e.pointerId === pointer) releaseStick(); });
}
window.addEventListener('blur', releaseStick);
setInterval(sendPtz, 120);
$('connectBtn').onclick = () => stream || connectionAttempt ? disconnect() : connect();
$('recordBtn').onclick = toggleRecord;
$('refreshBtn').onclick = () => stream ? detectControlInterface() : connect();
document.addEventListener('keydown', e => { if (e.key.toLowerCase() === 'r') toggleRecord(); });
