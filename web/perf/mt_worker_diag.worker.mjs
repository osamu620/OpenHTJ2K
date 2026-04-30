// Diagnostic worker that loads libopen_htj2k_mt_simd from inside a worker
// context and reports every error/message event from the inner pthread
// workers Emscripten spawns.  By default nested-worker errors come through
// the outer-worker console as "Uncaught [object Event]" with no useful
// detail; we monkey-patch `Worker` here to attach diagnostic listeners
// before each inner worker is bound to anything.

/* global self */

function send(level, msg) { self.postMessage({ level, msg: String(msg) }); }

// Catch anything that escapes try/catch.
self.addEventListener('error', (ev) => {
  send('err', `outer worker uncaught: ${ev.message} (${ev.filename}:${ev.lineno})`);
});
self.addEventListener('unhandledrejection', (ev) => {
  send('err', `outer worker unhandledrejection: ${ev.reason?.stack || ev.reason}`);
});

// Wrap the global Worker constructor so we see every inner Emscripten
// worker spawn AND its full error/message stream.
const OriginalWorker = self.Worker;
let innerSeq = 0;
self.Worker = new Proxy(OriginalWorker, {
  construct(target, args) {
    const id = ++innerSeq;
    const [url, opts] = args;
    send('info', `[inner #${id}] new Worker(${JSON.stringify(String(url))}, ${JSON.stringify(opts)})`);
    let inst;
    try {
      inst = new target(...args);
    } catch (e) {
      send('err', `[inner #${id}] constructor threw: ${e?.stack || e}`);
      throw e;
    }
    inst.addEventListener('error', (ev) => {
      send('err', `[inner #${id}] error event: ` +
        `message=${ev.message ?? '<undefined>'} ` +
        `filename=${ev.filename ?? '<undefined>'} ` +
        `lineno=${ev.lineno ?? '<undefined>'} ` +
        `colno=${ev.colno ?? '<undefined>'} ` +
        `error=${ev.error ? (ev.error.stack || ev.error) : '<no Error object>'}`);
    });
    inst.addEventListener('messageerror', (ev) => {
      send('err', `[inner #${id}] messageerror: ${ev.data}`);
    });
    // Wrap postMessage so we can see what commands Emscripten sends to
    // the inner pthreads — useful to confirm "load" is reached.
    const origPost = inst.postMessage.bind(inst);
    inst.postMessage = (data, transfer) => {
      try {
        const cmd = data?.cmd;
        send('info', `[inner #${id}] ← postMessage cmd=${cmd}`);
      } catch (_) {}
      return transfer ? origPost(data, transfer) : origPost(data);
    };
    inst.addEventListener('message', (ev) => {
      const cmd = ev.data?.cmd;
      send('info', `[inner #${id}] → message cmd=${cmd}`);
    });
    return inst;
  },
});

self.addEventListener('message', async ({ data }) => {
  if (data.type !== 'init') return;
  send('info', `worker location: ${self.location.href}`);
  send('info', `crossOriginIsolated: ${self.crossOriginIsolated}`);
  send('info', `SharedArrayBuffer available: ${typeof SharedArrayBuffer !== 'undefined'}`);
  send('info', `navigator.hardwareConcurrency: ${navigator.hardwareConcurrency}`);

  const wasmBase = '/wasm/';
  const factoryURL = new URL(`${wasmBase}libopen_htj2k_mt_simd.js`, self.location.href);
  send('info', `importing factory from: ${factoryURL.href}`);
  let factory;
  try {
    const mod = await import(factoryURL.href);
    factory = mod.default;
    send('ok', 'factory module imported');
  } catch (e) {
    send('err', `factory import failed: ${e?.stack || e}`);
    return;
  }

  send('info', 'calling factory({ locateFile, mainScriptUrlOrBlob })...');
  try {
    const M = await factory({
      locateFile: (path) => {
        const u = new URL(path, factoryURL.href).href;
        send('info', `[locateFile] ${path} → ${u}`);
        return u;
      },
      mainScriptUrlOrBlob: factoryURL.href,
      print:    (s) => send('info', `[Module.print] ${s}`),
      printErr: (s) => send('warn', `[Module.printErr] ${s}`),
    });
    send('ok', 'factory() resolved — module ready');
    send('info', `M.HEAPU8 byteLength: ${M.HEAPU8.byteLength}`);
    send('info', `M._malloc available: ${typeof M._malloc === 'function'}`);
  } catch (e) {
    send('err', `factory() rejected: ${e?.stack || e}`);
  }
});
