// Independent contract checks. This file is never copied into the agent project.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";
const project = path.resolve(process.argv[2]);
const stage = Number(process.argv[3] || 6),
  hidden = process.argv.includes("--acceptance");
const rows = [];
const exportsRequired = {
  random: ["createRng", "hashSeed"],
  config: ["validateConfig", "serializeConfig", "parseConfig"],
  simulation: [
    "createGame",
    "tick",
    "flap",
    "pause",
    "resume",
    "restart",
    "cloneState",
    "restoreState",
  ],
  input: ["createInput", "normalizePointer", "createTicker"],
  app: ["mountApp"],
  scores: ["createScoreStore"],
};
const load = async (name) => {
  const m = await import(
    pathToFileURL(path.join(project, "src", name + ".js")).href
  );
  for (const k of exportsRequired[name] || [])
    assert.equal(typeof m[k], "function", name + "." + k + " must be exported");
  return m;
};
const detached = (a, b) => {
  assert.deepEqual(a, b);
  assert.notEqual(a, b);
};
const check = async (stageRequired, name, fn, sealed = false) => {
  if (stageRequired > stage || (sealed && !hidden)) return;
  try {
    await fn();
    rows.push({ name, passed: true, sealed });
  } catch (e) {
    rows.push({
      name,
      passed: false,
      sealed,
      error: String(e.stack || e).slice(0, 2200),
    });
  }
};
await check(1, "rng exact xorshift", async () => {
  const { createRng } = await load("random");
  const r = createRng(1);
  assert.deepEqual(
    [r.uint32(), r.uint32(), r.uint32()],
    [270369, 67634689, 2647435461],
  );
});
await check(1, "rng zero and restore", async () => {
  const { createRng } = await load("random");
  const a = createRng(0),
    b = createRng(0);
  assert.equal(a.next(), b.next());
  const s = a.snapshot(),
    n = a.next();
  a.restore(s);
  assert.equal(a.next(), n);
  assert.throws(() => a.restore(0));
});
await check(1, "hash FNV-1a", async () => {
  const { hashSeed } = await load("random");
  assert.equal(hashSeed("hello"), 1335831723);
  assert.equal(hashSeed(""), 2166136261);
});
await check(
  1,
  "rng invalid seeds",
  async () => {
    const { createRng } = await load("random");
    for (const v of [-1, 1.5, 4294967296, NaN, "1", null])
      assert.throws(() => createRng(v));
  },
  true,
);
await check(
  1,
  "rng invalid restore atomic",
  async () => {
    const { createRng } = await load("random");
    const a = createRng(73),
      s = a.snapshot();
    for (const v of [null, NaN, -1, 0, "3", 1.2]) {
      assert.throws(() => a.restore(v));
      assert.equal(a.snapshot(), s);
    }
  },
  true,
);
await check(1, "config defaults and detached", async () => {
  const { DEFAULT_CONFIG, validateConfig } = await load("config");
  const a = validateConfig({ seed: 0 });
  assert.equal(a.seed, 0);
  assert.equal(a.width, 480);
  assert.equal(a.gapHeight, 180);
  assert(Object.isFrozen(DEFAULT_CONFIG));
  assert.notEqual(a, DEFAULT_CONFIG);
});
await check(1, "config JSON roundtrip", async () => {
  const { validateConfig, serializeConfig, parseConfig } = await load("config");
  const a = validateConfig({ width: 600, seed: 917 });
  assert.deepEqual(parseConfig(serializeConfig(a)), a);
  assert.equal(serializeConfig(a), serializeConfig({ ...a }));
});
await check(1, "config reject invalid", async () => {
  const { validateConfig } = await load("config");
  for (const v of [
    null,
    [],
    { extra: 1 },
    { gravity: NaN },
    { gapHeight: 700 },
    { birdX: 1 },
    { seed: 1.2 },
  ])
    assert.throws(() => validateConfig(v));
});
await check(
  1,
  "config coupled and prototype edge",
  async () => {
    const { validateConfig } = await load("config");
    for (const v of [
      { height: 200, groundHeight: 150 },
      { width: 200, birdX: 195 },
      { gapHeight: 60, height: 200, groundHeight: 120 },
    ])
      assert.throws(() => validateConfig(v));
    assert.equal(validateConfig(Object.create({ gravity: 1 })).gravity, 0.35);
    const a = { seed: 0, gravity: 1 };
    const s = JSON.stringify(a);
    validateConfig(a);
    assert.equal(JSON.stringify(a), s);
  },
  true,
);
await check(2, "game initial and pure flap", async () => {
  const s = await load("simulation");
  const a = s.createGame();
  assert.equal(a.phase, "ready");
  assert.equal(a.frame, 0);
  const before = JSON.stringify(a),
    b = s.flap(a);
  assert.equal(b.phase, "running");
  assert.equal(b.bird.vy, -6);
  assert.equal(JSON.stringify(a), before);
  assert.notEqual(a.config, b.config);
});
await check(2, "game fixed-step flight", async () => {
  const s = await load("simulation");
  const a = s.flap(s.createGame()),
    b = s.tick(a);
  assert.equal(b.frame, 1);
  assert.equal(b.bird.vy, -5.65);
  assert.equal(b.bird.y, 324.35);
  assert.equal(a.bird.y, 330);
});
await check(2, "game paused and restarted", async () => {
  const s = await load("simulation");
  let a = s.tick(s.flap(s.createGame({ seed: 42 })));
  const p = s.pause(a);
  assert.equal(p.phase, "paused");
  assert.deepEqual(s.tick(p, { flap: true }), p);
  assert.equal(s.resume(p).phase, "running");
  assert.deepEqual(s.restart(a), s.createGame({ seed: 42 }));
});
await check(2, "game seeded pipes", async () => {
  const s = await load("simulation");
  let a = s.flap(
      s.createGame({
        spawnEvery: 10,
        gapHeight: 600,
        gravity: 0.01,
        flapVelocity: -0.1,
      }),
    ),
    b = s.cloneState(a);
  for (let i = 0; i < 20; i++) {
    a = s.tick(a);
    b = s.tick(b);
  }
  assert.deepEqual(a, b);
  assert.equal(a.pipes.length, 2);
  assert.equal(a.pipes[0].id, 10);
  assert.equal(a.pipes[1].id, 20);
  assert.equal(a.pipes[1].x, 480);
});
await check(2, "game lower boundary and no points on crash", async () => {
  const s = await load("simulation");
  let a = s.flap(s.createGame());
  a.bird.y = 647;
  a.bird.vy = 2;
  a.pipes = [{ id: 1, x: 0, gapY: 20, passed: false }];
  const b = s.tick(a);
  assert.equal(b.phase, "gameover");
  assert.equal(b.score, 0);
});
await check(2, "game passing one pipe once", async () => {
  const s = await load("simulation");
  let a = s.flap(s.createGame());
  a.bird.vy = 0;
  a.pipes = [{ id: 1, x: 40, gapY: 240, passed: false }];
  let b = s.tick(a);
  assert.equal(b.score, 1);
  assert.equal(b.pipes[0].passed, true);
  b = s.tick(b);
  assert.equal(b.score, 1);
  assert.equal(a.pipes[0].passed, false);
});
await check(2, "game snapshot roundtrip", async () => {
  const s = await load("simulation");
  const a = s.tick(s.flap(s.createGame()));
  const b = s.restoreState(JSON.parse(JSON.stringify(a)));
  detached(a, b);
  assert.notEqual(a.bird, b.bird);
  assert.throws(() => s.restoreState({ ...a, phase: "bogus" }));
});
await check(
  2,
  "game pipe and bird collision edges",
  async () => {
    const s = await load("simulation");
    for (const y of [112, 268]) {
      const a = s.flap(s.createGame());
      a.bird.y = y;
      a.bird.vy = -a.config.gravity;
      a.pipes = [{ id: 5, x: 134, gapY: 100, passed: false }];
      const b = s.tick(a);
      assert.equal(b.phase, "gameover");
    }
    const a = s.flap(s.createGame());
    a.bird.y = 200;
    a.bird.vy = -a.config.gravity;
    a.pipes = [{ id: 5, x: 134, gapY: 100, passed: false }];
    assert.equal(s.tick(a).phase, "running");
  },
  true,
);
await check(
  2,
  "game top boundary inclusive",
  async () => {
    const s = await load("simulation");
    const a = s.flap(s.createGame());
    a.bird.y = 12;
    a.bird.vy = -a.config.gravity;
    assert.equal(s.tick(a).phase, "gameover");
  },
  true,
);
await check(
  2,
  "game strict passing edge",
  async () => {
    const s = await load("simulation");
    const a = s.flap(s.createGame());
    a.bird.vy = -a.config.gravity;
    a.pipes = [{ id: 1, x: 46, gapY: 240, passed: false }];
    const b = s.tick(a);
    assert.equal(
      b.pipes[0].x + b.config.pipeWidth,
      b.bird.x - b.config.birdRadius,
    );
    assert.equal(b.score, 0);
    assert.equal(s.tick(b).score, 1);
  },
  true,
);
await check(
  2,
  "game removes offscreen pipes",
  async () => {
    const s = await load("simulation");
    const a = s.flap(s.createGame());
    a.pipes = [{ id: 1, x: -64, gapY: 200, passed: true }];
    assert.deepEqual(s.tick(a).pipes, []);
  },
  true,
);
await check(
  2,
  "game restore rejects corrupt snapshots",
  async () => {
    const s = await load("simulation");
    const a = s.createGame();
    for (const b of [
      { ...a, frame: -1 },
      { ...a, rngState: 0 },
      { ...a, bird: { ...a.bird, y: NaN } },
      { ...a, pipes: [{ id: 1, x: 2, gapY: 3, passed: "yes" }] },
      {
        ...a,
        pipes: [
          { id: 1, x: 2, gapY: 3, passed: false },
          { id: 1, x: 4, gapY: 3, passed: false },
        ],
      },
    ])
      assert.throws(() => s.restoreState(b));
  },
  true,
);
await check(
  2,
  "game no-op transitions detached",
  async () => {
    const s = await load("simulation");
    const a = s.createGame();
    for (const b of [s.tick(a), s.pause(a), s.resume(a)]) {
      detached(a, b);
      assert.notEqual(a.bird, b.bird);
    }
    let p = s.pause(s.flap(a));
    assert.deepEqual(s.flap(p), p);
  },
  true,
);
await check(3, "input edge queue and release", async () => {
  const { createInput } = await load("input");
  const a = createInput();
  assert.equal(a.handle({ code: "Space" }), true);
  assert.equal(a.handle({ code: "Space" }), true);
  assert.deepEqual(a.consume(), ["flap"]);
  assert.deepEqual(a.consume(), []);
  assert.equal(a.release({ code: "Space" }), true);
  a.handle({ code: "Space" });
  a.handle({ code: "KeyP" });
  assert.deepEqual(a.consume(), ["flap", "pause"]);
});
await check(3, "input ignore repeat and disposal", async () => {
  const { createInput } = await load("input");
  const a = createInput();
  a.handle({ code: "Space", repeat: true });
  assert.deepEqual(a.consume(), []);
  a.handle({ code: "KeyR" });
  a.reset();
  assert.deepEqual(a.consume(), []);
  a.dispose();
  assert.equal(a.handle({ code: "Space" }), false);
  assert.equal(a.release({ code: "Space" }), false);
});
await check(3, "pointer mapping/clamp", async () => {
  const { normalizePointer } = await load("input");
  assert.deepEqual(
    normalizePointer(
      { clientX: 60, clientY: 120 },
      { left: 10, top: 20, width: 100, height: 200 },
      480,
      720,
    ),
    { x: 240, y: 360 },
  );
  assert.deepEqual(
    normalizePointer(
      { clientX: -1, clientY: 300 },
      { left: 10, top: 20, width: 100, height: 200 },
      480,
      720,
    ),
    { x: 0, y: 720 },
  );
});
await check(3, "ticker first frame and backlog cap", async () => {
  const { createTicker } = await load("input");
  let n = 0,
    dt = [];
  const a = createTicker(
    (v) => {
      n++;
      dt.push(v);
    },
    { hz: 50, maxSteps: 3 },
  );
  a.update(0);
  assert.equal(n, 0);
  a.update(40);
  assert.equal(n, 2);
  a.update(1040);
  assert.equal(n, 5);
  a.update(1060);
  assert.equal(n, 6);
  assert(dt.every((x) => x === 0.02));
});
await check(3, "ticker pause fresh base and invalid time", async () => {
  const { createTicker } = await load("input");
  let n = 0;
  const a = createTicker(() => n++, { hz: 50 });
  a.update(100);
  assert.throws(() => a.update(90));
  a.update(120);
  assert.equal(n, 1);
  a.pause();
  a.update(10000);
  a.resume();
  a.update(12000);
  assert.equal(n, 1);
  a.update(12020);
  assert.equal(n, 2);
});
await check(
  3,
  "input custom binding validation",
  async () => {
    const { createInput } = await load("input");
    for (const b of [
      { flap: ["A"], pause: ["A"], restart: ["R"] },
      { bogus: ["A"] },
      { flap: [4] },
    ])
      assert.throws(() => createInput(b));
    const a = createInput({
      flap: ["KeyF"],
      pause: ["KeyS"],
      restart: ["KeyN"],
    });
    assert.equal(a.handle({ code: "Space" }), false);
    a.handle({ code: "KeyF" });
    assert.deepEqual(a.consume(), ["flap"]);
  },
  true,
);
await check(
  3,
  "pointer bad dimensions and values",
  async () => {
    const { normalizePointer } = await load("input");
    for (const r of [
      { left: 0, top: 0, width: 0, height: 2 },
      { left: NaN, top: 0, width: 2, height: 2 },
    ])
      assert.throws(() =>
        normalizePointer({ clientX: 1, clientY: 1 }, r, 10, 10),
      );
    assert.throws(() =>
      normalizePointer(
        { clientX: Infinity, clientY: 1 },
        { left: 0, top: 0, width: 2, height: 2 },
        10,
        10,
      ),
    );
  },
  true,
);
await check(
  3,
  "ticker fractional remainder",
  async () => {
    const { createTicker } = await load("input");
    let n = 0;
    const a = createTicker(() => n++, { hz: 50, maxSteps: 5 });
    for (const t of [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]) a.update(t);
    assert.equal(n, 4);
    a.update(100);
    assert.equal(n, 5);
  },
  true,
);
await check(
  3,
  "ticker invalid options",
  async () => {
    const { createTicker } = await load("input");
    for (const opts of [
      { hz: 0 },
      { hz: Infinity },
      { hz: 241 },
      { maxSteps: 0 },
      { maxSteps: 1.5 },
    ])
      assert.throws(() => createTicker(() => {}, opts));
  },
  true,
);
await check(5, "app import has no DOM side effects", async () => {
  const a = await load("app");
  assert.equal(typeof a.mountApp, "function");
});

const memoryStorage = (initial = null) => {
  let text = initial;
  return {getItem: () => text, setItem: (_key, value) => {text = value;}, removeItem: () => {text = null;}};
};
await check(6, "scores ordering detached lists and reload", async () => {
  const {createScoreStore} = await load("scores"), storage = memoryStorage();
  const store = createScoreStore(storage), input = {name:" Alice ",score:5,frames:300,seed:12345};
  store.record(input); input.name = "changed";
  store.record({name:"Bob",score:12,frames:400,seed:2});
  store.record({name:"Cara",score:12,frames:250,seed:3});
  const expected = [{name:"Cara",score:12,frames:250,seed:3},{name:"Bob",score:12,frames:400,seed:2},{name:"Alice",score:5,frames:300,seed:12345}];
  assert.deepEqual(store.list(), expected);
  const copied = store.list(); copied[0].score = 999;
  assert.deepEqual(store.list(),expected);
  assert.deepEqual(createScoreStore(storage).list(),expected);
});
await check(6, "scores best ten tie order and clear", async () => {
  const {createScoreStore} = await load("scores"), storage = memoryStorage(), store = createScoreStore(storage);
  for(let i=0;i<12;i++) store.record({name:"P"+i,score:i,frames:10,seed:i});
  assert.equal(store.list().length,10); assert.equal(store.list()[0].score,11); assert.equal(store.list()[9].score,2);
  store.clear(); assert.deepEqual(createScoreStore(storage).list(),[]);
  store.record({name:"first",score:3,frames:9,seed:1}); store.record({name:"second",score:3,frames:9,seed:1});
  assert.deepEqual(store.list().map(x=>x.name),["first","second"]);
});
await check(6, "scores invalid entry leaves list unchanged", async () => {
  const {createScoreStore} = await load("scores"), store=createScoreStore(memoryStorage());
  const valid={name:"Player",score:1,frames:2,seed:3};store.record(valid);
  for(const bad of [{...valid,name:" "},{...valid,score:1.5},{...valid,frames:-1},{...valid,seed:4294967296},{...valid,score:NaN}]) assert.throws(()=>store.record(bad));
  assert.deepEqual(store.list(),[valid]);
},true);
await check(6, "scores corrupt persisted data is inert", async () => {
  const {createScoreStore} = await load("scores");
  for(const value of ["not json",JSON.stringify({version:2,entries:[]}),JSON.stringify({version:1,entries:[{name:"X",score:-1,frames:0,seed:1}]})]) assert.deepEqual(createScoreStore(memoryStorage(value)).list(),[]);
},true);
await check(6, "scores storage failure is reported", async () => {
  const {createScoreStore} = await load("scores");
  assert.throws(()=>createScoreStore({getItem(){throw Error("unavailable");},setItem(){},removeItem(){}}));
  const store=createScoreStore({getItem(){return null;},setItem(){throw Error("full");},removeItem(){throw Error("full");}});
  assert.throws(()=>store.record({name:"Player",score:1,frames:2,seed:3}));
});

console.log(
  JSON.stringify({
    mode: hidden ? "acceptance" : "visible",
    checks: rows,
    passed: rows.filter((x) => x.passed).length,
    total: rows.length,
  }),
);
process.exitCode = rows.every((x) => x.passed) ? 0 : 1;
