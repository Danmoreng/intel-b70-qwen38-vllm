// Supplemental visual review, run only after scored inference has finished.
// These probes are separate from the frozen 45-case acceptance score.
import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";
import http from "node:http";

const project = await fs.realpath(process.argv[2]);
const output = path.resolve(process.argv[3]);
await fs.mkdir(output, { recursive: true });
const errors = [],
  remoteRequests = [];
const server = http.createServer(async (req, res) => {
  try {
    const rel = decodeURIComponent(
      new URL(req.url, "http://localhost").pathname,
    );
    if (rel === "/__review__") {
      res.setHeader("Content-Type", "text/html");
      res.end(
        '<!doctype html><html><body style="margin:0;background:#ddd"></body></html>',
      );
      return;
    }
    const file = await fs.realpath(
      path.resolve(project, "." + (rel === "/" ? "/index.html" : rel)),
    );
    if (!file.startsWith(project + path.sep)) {
      res.writeHead(403).end();
      return;
    }
    const types = {
      ".js": "text/javascript",
      ".html": "text/html",
      ".css": "text/css",
      ".json": "application/json",
    };
    res.setHeader("Content-Type", types[path.extname(file)] || "text/plain");
    res.end(await fs.readFile(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = "http://127.0.0.1:" + server.address().port;
let browser;
const result = {
  purpose:
    "Supplemental post-run visual review; excluded from frozen functional score and inference timing",
  project,
  scene: {
    width: 480,
    height: 720,
    bird: { x: 120, y: 330, vy: 0 },
    pipes: [
      { id: 100, x: 200, gapY: 200, passed: false },
      { id: 101, x: 400, gapY: 360, passed: false },
    ],
  },
};
try {
  browser = await chromium.launch({
    headless: true,
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({
    viewport: { width: 480, height: 720 },
    deviceScaleFactor: 1,
  });
  await page.route("**/*", async (route) => {
    const url = route.request().url();
    if (
      url.startsWith(base + "/") ||
      url.startsWith("data:") ||
      url.startsWith("blob:")
    )
      await route.continue();
    else {
      remoteRequests.push(url);
      await route.abort();
    }
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(base + "/__review__");
  result.render = await page.evaluate(async (scene) => {
    const { createGame } = await import("/src/simulation.js");
    const { createRenderer } = await import("/src/renderer.js");
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const r = createRenderer(canvas, { reducedMotion: true });
    r.resize(scene.width, scene.height, 1);
    const gl = canvas.getContext("webgl2");
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = ext
      ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER);
    if (!/swiftshader/i.test(gpu))
      throw Error("Expected software renderer: " + gpu);
    const state = createGame();
    state.phase = "running";
    state.frame = 120;
    state.bird = { ...state.bird, ...scene.bird };
    state.pipes = scene.pipes;
    const positions = {
      pipeTop: [232, 50],
      pipeBottom: [232, 600],
      pipeGap: [232, 290],
      bird: [120, 330],
    };
    const sample = () =>
      Object.fromEntries(
        Object.entries(positions).map(([name, [x, y]]) => {
          const rgba = new Uint8Array(4);
          gl.readPixels(
            x,
            scene.height - 1 - y,
            1,
            1,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            rgba,
          );
          return [name, Array.from(rgba)];
        }),
      );
    r.render(
      { ...state, pipes: [], bird: { ...state.bird, x: -1000 } },
      { reducedMotion: true },
    );
    const empty = sample();
    const before = JSON.stringify(state);
    r.render(state, { reducedMotion: true });
    const drawn = sample();
    const equal = (name) =>
      JSON.stringify(empty[name]) === JSON.stringify(drawn[name]);
    return {
      gpu,
      empty,
      drawn,
      glError: gl.getError(),
      stateUnchanged: before === JSON.stringify(state),
      checks: {
        topPipeVisible: !equal("pipeTop"),
        bottomPipeVisible: !equal("pipeBottom"),
        gapRemainsOpen: equal("pipeGap"),
        birdVisible: !equal("bird"),
      },
    };
  }, result.scene);
  await page.screenshot({ path: path.join(output, "controlled-pipes.png") });
  await page.goto(base + "/__review__");
  await page.setViewportSize({ width: 600, height: 900 });
  result.play = await page.evaluate(async () => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/style.css";
    document.head.append(link);
    const root = document.createElement("main");
    root.id = "app";
    document.body.append(root);
    const { mountApp } = await import("/src/app.js");
    window.__reviewApp = mountApp(root, { autoStart: false });
    window.__reviewActions = [];
    window.__reviewAdvance = (count) => {
      for (let i = 0; i < count; i++) {
        const s = window.__reviewApp.getState();
        if (s.phase !== "running") break;
        const next = s.pipes.find(
          (p) => p.x + s.config.pipeWidth >= s.bird.x - s.config.birdRadius,
        );
        const target = next
          ? next.gapY + s.config.gapHeight / 2
          : (s.config.height - s.config.groundHeight) / 2;
        if (s.bird.y > target + 10 && s.bird.vy >= 0) {
          window.__reviewActions.push(s.frame);
          window.__reviewApp.flap();
        }
        window.__reviewApp.step(1);
      }
    };
    window.__reviewApp.start();
    window.__reviewAdvance(450);
    return { sceneAt450: window.__reviewApp.getState() };
  });
  await page.waitForTimeout(100);
  await page.screenshot({
    path: path.join(output, "played-game.png"),
    fullPage: true,
  });
  result.play.final = await page.evaluate(() => {
    window.__reviewAdvance(1050);
    return {
      state: window.__reviewApp.getState(),
      flapFrames: window.__reviewActions,
    };
  });
  result.play.passed =
    result.play.final.state.phase === "running" &&
    result.play.final.state.frame === 1500 &&
    result.play.final.state.score >= 3;
  await page.evaluate(() => window.__reviewApp.destroy());
} catch (e) {
  result.error = String(e);
} finally {
  if (browser) await browser.close();
  await new Promise((r) => server.close(r));
}
result.pageErrors = errors;
result.remoteRequests = remoteRequests;
result.passed =
  !!result.render &&
  Object.values(result.render.checks).every(Boolean) &&
  result.render.glError === 0 &&
  result.render.stateUnchanged &&
  result.play?.passed === true &&
  !errors.length &&
  !remoteRequests.length;
await fs.writeFile(
  path.join(output, "visual-review.json"),
  JSON.stringify(result, null, 2) + "\n",
);
console.log(JSON.stringify(result));
