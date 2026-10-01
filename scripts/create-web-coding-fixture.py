#!/usr/bin/env python3
"""Author the explicitly scoped small Flappy Bird calibration fixture."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/web-coding-fixture/pilot-v1"

CONTRACTS = {
    "random": "Export createRng(seed), hashSeed(text). Seeds are unsigned 32-bit integers, including "
    "zero; reject other inputs. createRng returns {next,uint32,snapshot,restore}. uint32 "
    "advances xorshift32: x ^= x<<13; x ^= x>>>17; x ^= x<<5, truncating unsigned after the "
    "sequence. Zero initializes to 0x6d2b79f5. next() = uint32()/4294967296. snapshot "
    "returns the unsigned current state; restore accepts only a nonzero uint32 and returns "
    "nothing. Invalid restore cannot change state. hashSeed is FNV-1a over JavaScript "
    "UTF-16 code units, initial 2166136261, Math.imul(hash ^ codeUnit,16777619), final "
    "unsigned. Reproducibility must not depend on Date.now or Math.random. Return separate "
    "stateful instances.",
    "config": "Export DEFAULT_CONFIG (deep frozen), validateConfig(value), serializeConfig(value), "
    "parseConfig(text). Configuration has exactly the known keys: "
    "width=480,height=720,birdX=120,birdRadius=12,gravity=0.35,flapVelocity=-6,pipeSpeed=2,pipeWidth=64,gapHeight=180,spawnEvery=100,groundHeight=60,seed=12345. "
    "validateConfig accepts an object containing a subset of keys and returns a new "
    "complete object with defaults. No unknown keys; no null, arrays, inherited values, NaN "
    "or infinity. width,height positive integer 200..2048; birdX finite, >birdRadius and "
    "<width-birdRadius; birdRadius finite 4..40; gravity finite 0.01..2; flapVelocity "
    "finite -20..-0.1; pipeSpeed finite 0.1..10; pipeWidth integer 10..200; gapHeight "
    "integer 60..height-groundHeight-40; spawnEvery integer 10..1000; groundHeight integer "
    "0..200 with height-groundHeight>100; seed uint32. Joint constraints are checked after "
    "defaults. serializeConfig returns compact JSON with keys in DEFAULT_CONFIG order. "
    "parseConfig parses JSON then validates. Never mutate the input.",
    "simulation": "Export createGame(config={}), tick(state,input={}), flap(state), pause(state), "
    "resume(state), restart(state), cloneState(state), restoreState(snapshot). All "
    "transitions are pure and return detached objects: never mutate state, input, or "
    "config. State: "
    "{version:1,config,phase:'ready'|'running'|'paused'|'gameover',frame:0,score:0,rngState,bird:{x:config.birdX,y:(height-groundHeight)/2,vy:0},pipes:[]}. "
    "createGame uses validated config and RNG initialized by seed. flap on ready "
    "changes phase to running and sets vy=flapVelocity; on running sets vy likewise; on "
    "paused/gameover is a detached no-op. tick on nonrunning returns detached unchanged "
    "state. Running tick increments frame, applies flap if input.flap===true, then "
    "vy+=gravity, y+=vy. Existing pipes x-=pipeSpeed. Every spawnEvery frames append "
    "{id:frame,x:width,gapY,passed:false}; "
    "gapY=20+floor(rng.next()*(height-groundHeight-gapHeight-40+1)); save rng state. A "
    "pipe collision uses bird AABB inclusive edges: overlap when bird.x+r>=pipe.x && "
    "bird.x-r<=pipe.x+pipeWidth; collide if bird.y-r<=gapY or bird.y+r>=gapY+gapHeight. "
    "Boundary collision when y-r<=0 or y+r>=height-groundHeight. Each unpassed pipe "
    "whose right edge is strictly <bird.x-birdRadius becomes passed and adds one score; "
    "remove pipes whose right edge<0. A collision changes phase to gameover; do not "
    "award points for the colliding frame. pause only running -> paused; resume only "
    "paused -> running. restart resets using same config. cloneState deeply clones. "
    "restoreState validates complete shape, uint32 rngState nonzero, nonnegative "
    "integer frame/score, config, legal phase, finite bird values and unique "
    "nonnegative integer pipe IDs, finite x/gapY and boolean passed; rejects invalid "
    "atomically and returns detached. Valid snapshots may contain manually placed pipes "
    "and any finite bird velocity, used by replay/test tooling. No wall clock or DOM "
    "dependencies.",
    "input": "Export "
    "createInput(bindings={flap:['Space','ArrowUp'],pause:['KeyP'],restart:['KeyR']}), "
    "normalizePointer(event,rect,width,height), "
    "createTicker(step,options={hz:60,maxSteps:5}). Input object "
    "{handle,release,consume,reset,dispose}; handle({code,repeat?,type?}) uses "
    "type='keydown' default; keyup delegates release. Only known bindings, arrays of "
    "nonempty unique strings across actions; recognized nonrepeat keydown emits an edge "
    "action once while held, returns true; repeats/second held keydown recognized returns "
    "true without queueing; keyup returns recognized boolean. consume returns queued actions "
    "in order and empties; reset clears held and queue; dispose then future handles/releases "
    "return false and consume=[]; no global DOM listeners. normalizePointer maps finite "
    "clientX/clientY relative to finite rect {left,top,width,height} with positive "
    "dimensions to logical coordinates clamped0..width/height; logical width/height "
    "positive; malformed throws. Ticker returns {update,pause,resume,reset}; hz positive "
    "finite<=240, maxSteps integer1..100; update(timestampMs) first timestamp initializes, "
    "then computes elapsed fixed dt=1000/hz and invokes step(dt/1000) up to maxSteps; drops "
    "excess backlog, retains fractional remainder; decreasing timestamp rejects atomically; "
    "pause prevents callbacks and clears elapsed base; resume starts fresh base; reset fresh "
    "base. Reject nonfinite timestamps. No requestAnimationFrame internally.",
    "renderer": "Export createRenderer(canvas,options={}). Require actual WebGL2 via "
    "canvas.getContext('webgl2'); no Canvas2D/game third-party libraries. Throw clear "
    "Error if unavailable. Renderer {resize,render,destroy,info}. Compile vertex/fragment "
    "shaders, draw procedural bird, pipes, sky, ground, no network/assets required. "
    "resize(width,height,dpr=1) requires finite positive values, DPR<=4, sets drawing "
    "buffer to round(width*dpr),round(height*dpr), CSS size in px, updates viewport; "
    "logical state scales to canvas dimensions. "
    "render(state,{highContrast=false,reducedMotion=false}={}) draws enough "
    "distinguishable colors/regions, clears every frame, and handles "
    "ready/running/paused/gameover. No mutation of state. info returns "
    "{backend:'webgl2',width,height,dpr,contextLost:boolean,destroyed:boolean}. Listen "
    "for webglcontextlost, preventDefault and stop drawing; on restore rebuild resources "
    "and resume. destroy is idempotent, deletes buffers/programs/shaders and removes "
    "listeners; afterwards render and resize throw. Options highContrast changes palette. "
    "No animation loop inside renderer; app owns scheduling. Testable with software "
    "WebGL; do not use WebGPU fallback or DOM overlays instead of rendering gameplay.",
    "app": "Export mountApp(root,options={}) from src/app.js, safely importable in Node (no top-level "
    "DOM). Root is a DOM element. options {autoStart?:true}; autoStart false disables "
    "requestAnimationFrame for tests. Returns "
    "{start,step,flap,pause,resume,restart,getState,destroy}. Integrate simulation, renderer "
    'and input. Markup has exactly one canvas[data-testid="game-canvas"], '
    'button[data-testid="start"], button[data-testid="pause"], button[data-testid="restart"], '
    'element[data-testid="score"], element[data-testid="status"], element[data-testid="live"] '
    "with role=status and aria-live=polite. Buttons have accessible labels. Initial ready "
    "game; start() moves to running; flap() starts or flaps; step(count=1) advances count "
    "integer0..10000 ticks while running and renders/updates UI; pause/resume preserve state; "
    "restart resets same config to ready. getState returns detached snapshots. Keyboard "
    "Space/ArrowUp flap, KeyP toggles pause, KeyR restart; pointerdown canvas flaps; ignore "
    "repeated held keydowns. Ignore keyboard while input/textarea focused. Visibility hidden "
    "pauses. Responsive canvas resize, DPR capped<=4. destroy removes all listeners, cancels "
    "rAF, destroys renderer and is idempotent. Subsequent commands except getState throw. "
    "index.html loads main.js module calling mountApp on #app; CSS supports mobile layout, "
    "keyboard focus and reduced motion. No third-party libraries, replay, level editor, "
    "campaign or backend. Add a short USER_GUIDE.md with controls, deterministic seed and "
    "static-server instructions. The objective is a small working game.",
}

STAGES = [
    (
        "01-foundation",
        "random config",
        "Implement seeded random and validated game configuration.",
    ),
    (
        "02-physics",
        "simulation",
        "Implement deterministic fixed-step flight, seeded pipes, collision, scoring and pure state "
        "transitions. Include pause/resume/restart and detached snapshots.",
    ),
    (
        "03-controls",
        "input",
        "Implement edge-triggered keyboard input, pointer mapping and bounded fixed-step ticker.",
    ),
    (
        "04-webgl",
        "renderer",
        "Implement procedural WebGL2 bird/pipes/ground/sky, resize/DPR and context loss/destroy "
        "lifecycle. Use browser_test to verify rendering.",
    ),
    (
        "05-game",
        "app",
        "Build a small playable responsive app with keyboard/touch, start/pause/restart, score/status "
        "and accessible controls. Integrate the physics and renderer and verify in the browser.",
    ),
    (
        "06-release",
        "random config simulation input renderer app",
        "Check all contracts and the real browser game. Inspect and fix any remaining issues, run all "
        "visible tests plus browser checks, write the short user guide. Stop when the small game is "
        "complete; do not add editor, replay or unrelated features.",
    ),
]


def main():
    if (ROOT / "manifest.json").exists():
        raise SystemExit(
            "fixture is frozen; author a new version rather than overwrite"
        )
    project = ROOT / "project"
    (project / "src").mkdir(parents=True, exist_ok=True)
    (project / "specs").mkdir(exist_ok=True)
    (project / "tests").mkdir(exist_ok=True)
    (project / "package.json").write_text(
        json.dumps(
            {
                "name": "flappylab-benchmark",
                "private": True,
                "type": "module",
                "version": "1.0.0",
                "scripts": {"test": "node --test tests/*.test.js"},
            },
            indent=2,
        )
        + "\n"
    )
    (project / "README.md").write_text(
        """# FlappyLab\n\nBuild a complete offline, dependency-free WebGL2 Flappy Bird laboratory from this starter. Use JavaScript ES modules. Node tests and an external software-rendered browser harness are provided by the benchmark. The game must have no third-party runtime dependencies.\n\nEach specs/*.md file is an exact public API contract. Implement all exports described there, retaining contracts from earlier stages. Pure modules must import safely in Node and must never use a wall clock or mutate caller inputs. DOM/audio/WebGL initialization must be lazy. Tests and specs are protected; place your own regression tests in tests/*.test.js. You can edit src/**, index.html, main.js, style.css and USER_GUIDE.md.\n\nStages are revealed one at a time in one conversation, without compaction. Finish each stage with a brief status after using the available tools to inspect, implement and test. There is no duration or token quota; implement the actual requested behavior efficiently. Tests are not a substitute for a usable app. The final release is also checked against independent acceptance cases unavailable to the agent.\n\nRun with a static HTTP server, e.g. python3 -m http.server 8765. Entry point index.html -> main.js -> src/app.js. No bundler, remote CDN, downloaded media or build step.\n"""
    )
    for name, contract in CONTRACTS.items():
        (project / "specs" / (name + ".md")).write_text(
            "# " + name + " public API contract\n\n" + contract + "\n"
        )
        (project / "src" / (name + ".js")).write_text(
            "// Implement the exact public API in specs/"
            + name
            + ".md.\n// This starter intentionally has no implementation.\nexport {};\n"
        )
    (project / "index.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlappyLab</title><link rel="stylesheet" href="style.css"></head><body><main id="app"></main><script type="module" src="main.js"></script></body></html>\n'
    )
    (project / "main.js").write_text(
        "import {mountApp} from './src/app.js';\nmountApp(document.querySelector('#app'));\n"
    )
    (project / "style.css").write_text(
        "body { margin:0; font-family:system-ui,sans-serif; }\n"
    )
    (project / "tests" / "starter.test.js").write_text(
        "import test from 'node:test';\nimport assert from 'node:assert/strict';\ntest('starter package loads',()=>assert.ok(true));\n"
    )
    tasks = []
    for stage_id, modules, instruction in STAGES:
        tasks.append(
            {
                "id": stage_id,
                "modules": modules.split(),
                "instruction": instruction
                + " Read the applicable specs and current source before editing. After changes, read back the changed modules to audit interactions, add focused regression tests, and run_tests. Browser stages must also use browser_test. Do not change protected specs or acceptance harness. All prior functionality must remain working.",
            }
        )
    (ROOT / "tasks.json").write_text(
        json.dumps(
            {"fixture_id": "flappylab-webgl2-pilot-v1", "tasks": tasks}, indent=2
        )
        + "\n"
    )
    print("authored", len(tasks), "stages and", len(CONTRACTS), "module contracts")


if __name__ == "__main__":
    main()
