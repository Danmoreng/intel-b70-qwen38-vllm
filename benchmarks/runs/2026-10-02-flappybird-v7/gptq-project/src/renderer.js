// Procedural WebGL2 renderer, per specs/renderer.md.
// Logical pixels use a top-left origin; each vertex is converted to clip
// space exactly once, in the vertex shader, using the logical config size.

const VERT = `#version 300 es
layout(location = 0) in vec2 aPos;
uniform vec2 uSize;
void main() {
  vec2 c = aPos / uSize;
  gl_Position = vec4(2.0 * c.x - 1.0, 1.0 - 2.0 * c.y, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }`;

const PALETTES = {
  normal: {
    sky: [0.45, 0.76, 0.98, 1],
    sun: [1, 0.92, 0.55, 1],
    cloud: [1, 1, 1, 0.9],
    ground: [0.55, 0.38, 0.2, 1],
    grass: [0.28, 0.62, 0.25, 1],
    dirt: [0.4, 0.27, 0.13, 1],
    pipe: [0.18, 0.62, 0.24, 1],
    pipeLip: [0.12, 0.46, 0.16, 1],
    bird: [1, 0.8, 0.15, 1],
    wing: [0.9, 0.55, 0.1, 1],
    eye: [1, 1, 1, 1],
    pupil: [0.05, 0.05, 0.05, 1],
    beak: [0.95, 0.45, 0.05, 1],
  },
  high: {
    sky: [0, 0, 0, 1],
    sun: [1, 1, 1, 1],
    cloud: [0.7, 0.7, 0.7, 1],
    ground: [0.85, 0.85, 0.85, 1],
    grass: [1, 1, 1, 1],
    dirt: [0.25, 0.25, 0.25, 1],
    pipe: [1, 1, 1, 1],
    pipeLip: [0.5, 0.5, 0.5, 1],
    bird: [1, 0.25, 0.25, 1],
    wing: [1, 0.6, 0.2, 1],
    eye: [1, 1, 1, 1],
    pupil: [0, 0, 0, 1],
    beak: [1, 1, 0, 1],
  },
};

function rectVerts(x, y, w, h) {
  return [x, y, x + w, y, x, y + h, x, y + h, x + w, y, x + w, y + h];
}

function circleVerts(cx, cy, r, n) {
  const verts = [];
  for (let i = 0; i < n; i += 1) {
    const a0 = (i / n) * Math.PI * 2;
    const a1 = ((i + 1) / n) * Math.PI * 2;
    verts.push(cx, cy,
      cx + Math.cos(a0) * r, cy + Math.sin(a0) * r,
      cx + Math.cos(a1) * r, cy + Math.sin(a1) * r);
  }
  return verts;
}

function triVerts(x0, y0, x1, y1, x2, y2) {
  return [x0, y0, x1, y1, x2, y2];
}

export function createRenderer(canvas, options = {}) {
  if (canvas === null || typeof canvas !== 'object' ||
    typeof canvas.getContext !== 'function') {
    throw new TypeError('createRenderer requires a canvas element');
  }
  const gl = canvas.getContext('webgl2');
  if (gl === null) {
    throw new Error('WebGL2 is not available on this canvas');
  }

  const meta = {
    width: 0,
    height: 0,
    dpr: 1,
    contextLost: false,
    destroyed: false,
  };
  let program = null;
  let vao = null;
  let vbo = null;

  function info() {
    return {
      backend: 'webgl2',
      width: meta.width,
      height: meta.height,
      dpr: meta.dpr,
      contextLost: meta.contextLost,
      destroyed: meta.destroyed,
    };
  }

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`shader compilation failed: ${log}`);
    }
    return shader;
  }

  function releaseGpu() {
    if (vao !== null) {
      gl.deleteVertexArray(vao);
      vao = null;
    }
    if (vbo !== null) {
      gl.deleteBuffer(vbo);
      vbo = null;
    }
    if (program !== null) {
      gl.deleteProgram(program);
      program = null;
    }
  }

  function build() {
    releaseGpu();
    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG);
    program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.bindAttribLocation(program, 0, 'aPos');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`shader link failed: ${log}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    gl.useProgram(program);
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  }

  build();

  function applyViewport() {
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  const onLost = (e) => {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault();
    }
    meta.contextLost = true;
  };
  const onRestored = () => {
    if (!meta.contextLost) {
      return;
    }
    build();
    meta.contextLost = false;
    applyViewport();
  };
  canvas.addEventListener('webglcontextlost', onLost, false);
  canvas.addEventListener('webglcontextrestored', onRestored, false);

  function resize(width, height, dpr = 1) {
    if (meta.destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0) {
      throw new RangeError('resize width must be finite and positive');
    }
    if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) {
      throw new RangeError('resize height must be finite and positive');
    }
    if (typeof dpr !== 'number' || !Number.isFinite(dpr) || dpr <= 0 || dpr > 4) {
      throw new RangeError('resize dpr must be finite, positive and <= 4');
    }
    meta.width = width;
    meta.height = height;
    meta.dpr = dpr;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (!meta.contextLost) {
      applyViewport();
    }
  }

  function draw(color, verts) {
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.uniform4fv(gl.getUniformLocation(program, 'uColor'), color);
    gl.drawArrays(gl.TRIANGLES, 0, verts.length / 2);
  }

  function drawScene(state, palette) {
    const {width: W, height: H} = state.config;
    const ground = state.config.groundHeight;
    const gapH = state.config.gapHeight;
    const pipeW = state.config.pipeWidth;
    const bird = state.bird;
    const r = state.config.birdRadius;
    gl.uniform2f(gl.getUniformLocation(program, 'uSize'), W, H);

    draw(palette.sky, rectVerts(0, 0, W, H));
    draw(palette.sun, circleVerts(W - 64, 64, 26, 12));
    draw(palette.cloud, rectVerts(40, 110, 90, 16));
    draw(palette.cloud, rectVerts(200, 170, 70, 14));

    draw(palette.ground, rectVerts(0, H - ground, W, ground));
    draw(palette.grass, rectVerts(0, H - ground, W, Math.min(10, ground)));
    if (ground > 14) {
      for (let x = 8; x < W - 8; x += 32) {
        draw(palette.dirt, rectVerts(x, H - ground + 16, 14, 7));
      }
    }

    for (const pipe of state.pipes) {
      const topH = pipe.gapY;
      const bottomY = pipe.gapY + gapH;
      const bottomH = H - ground - bottomY;
      draw(palette.pipe, rectVerts(pipe.x, 0, pipeW, topH));
      draw(palette.pipeLip, rectVerts(pipe.x - 3, Math.max(0, topH - 14), pipeW + 6, 14));
      if (bottomH > 0) {
        draw(palette.pipe, rectVerts(pipe.x, bottomY, pipeW, bottomH));
        draw(palette.pipeLip, rectVerts(pipe.x - 3, bottomY, pipeW + 6, 14));
      }
    }

    draw(palette.bird, circleVerts(bird.x, bird.y, r, 12));
    draw(palette.wing, circleVerts(bird.x - r * 0.35, bird.y + r * 0.2, r * 0.5, 10));
    draw(palette.beak, triVerts(bird.x + r * 0.5, bird.y - r * 0.05,
      bird.x + r * 1.15, bird.y + r * 0.18,
      bird.x + r * 0.5, bird.y + r * 0.45));
    draw(palette.eye, circleVerts(bird.x + r * 0.45, bird.y - r * 0.35, r * 0.38, 10));
    draw(palette.pupil, circleVerts(bird.x + r * 0.55, bird.y - r * 0.35, r * 0.16, 8));
  }

  function render(state, flags = {}) {
    if (meta.destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (meta.contextLost || program === null) {
      return;
    }
    const highContrast = flags !== undefined && flags !== null &&
      flags.highContrast === true;
    const palette = highContrast ? PALETTES.high : PALETTES.normal;
    gl.clearColor(palette.sky[0], palette.sky[1], palette.sky[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    drawScene(state, palette);
  }

  let destroyed = false;

  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    meta.destroyed = true;
    canvas.removeEventListener('webglcontextlost', onLost, false);
    canvas.removeEventListener('webglcontextrestored', onRestored, false);
    releaseGpu();
  }

  return {
    resize,
    render,
    destroy,
    info,
  };
}
