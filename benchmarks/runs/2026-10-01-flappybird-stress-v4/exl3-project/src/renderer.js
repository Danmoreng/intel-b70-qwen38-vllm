// Public API contract: specs/renderer.md
const VERT_SRC = `
attribute vec2 aPos;
attribute vec4 aColor;
uniform float uScale;
uniform vec2 uOffset;
varying vec4 vColor;
void main() {
  vColor = aColor;
  gl_Position = vec4(uOffset + aPos * uScale, 0.0, 1.0);
}
`;

const FRAG_SRC = `
precision mediump float;
varying vec4 vColor;
void main() {
  gl_FragColor = vColor;
}
`;

const PALETTE = {
  sky: [0.52, 0.8, 0.92],
  sun: [1, 0.85, 0.35],
  cloud: [0.97, 0.98, 1],
  ground: [0.76, 0.6, 0.32],
  grass: [0.45, 0.75, 0.35],
  pipe: [0.24, 0.66, 0.28],
  bird: [0.98, 0.8, 0.16],
  eye: [1, 1, 1],
  pausedOverlay: [0, 0, 0, 0.4],
  gameoverOverlay: [0.85, 0.15, 0.15, 0.35]
};

const HIGH_CONTRAST_PALETTE = {
  sky: [0, 0, 0],
  sun: [1, 1, 0],
  cloud: [1, 1, 1],
  ground: [0.6, 0.6, 0.6],
  grass: [0, 1, 0],
  pipe: [0, 1, 1],
  bird: [1, 0, 1],
  eye: [0, 0, 0],
  pausedOverlay: [1, 1, 1, 0.5],
  gameoverOverlay: [1, 1, 0, 0.4]
};

function pushRect(arr, x0, y0, x1, y1, color) {
  const [r, g, b, a = 1] = color;
  arr.push(
    x0, y0, r, g, b, a,
    x1, y0, r, g, b, a,
    x1, y1, r, g, b, a,
    x0, y0, r, g, b, a,
    x1, y1, r, g, b, a,
    x0, y1, r, g, b, a
  );
}

function pushCircle(arr, cx, cy, radius, color, segments = 24) {
  const [r, g, b] = color;
  for (let i = 0; i < segments; i += 1) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    arr.push(
      cx, cy, r, g, b, 1,
      cx + Math.cos(a0) * radius, cy + Math.sin(a0) * radius, r, g, b, 1,
      cx + Math.cos(a1) * radius, cy + Math.sin(a1) * radius, r, g, b, 1
    );
  }
}

export function createRenderer(canvas, options = {}) {
  if (typeof canvas !== 'object' || canvas === null || typeof canvas.getContext !== 'function') {
    throw new Error('createRenderer requires a canvas element');
  }
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 is not available');
  }

  let destroyed = false;
  let contextLost = false;
  let program = null;
  let vao = null;
  let buffer = null;
  let uScaleLoc = null;
  let uOffsetLoc = null;
  let width = canvas.width || 300;
  let height = canvas.height || 150;
  let dpr = 1;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`shader compile failed: ${log}`);
    }
    return shader;
  }

  function buildResources() {
    const vs = compileShader(gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);
    program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      program = null;
      throw new Error(`program link failed: ${log}`);
    }
    uScaleLoc = gl.getUniformLocation(program, 'uScale');
    uOffsetLoc = gl.getUniformLocation(program, 'uOffset');
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const aPos = gl.getAttribLocation(program, 'aPos');
    const aColor = gl.getAttribLocation(program, 'aColor');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(aColor);
    gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, 24, 8);
    gl.bindVertexArray(null);
  }

  function deleteResources() {
    if (buffer) gl.deleteBuffer(buffer);
    if (vao) gl.deleteVertexArray(vao);
    if (program) gl.deleteProgram(program);
    buffer = null;
    vao = null;
    program = null;
    uScaleLoc = null;
    uOffsetLoc = null;
  }

  const onContextLost = (event) => {
    event.preventDefault();
    contextLost = true;
  };

  const onContextRestored = () => {
    contextLost = false;
    if (!destroyed) {
      deleteResources();
      buildResources();
      if (canvas.width > 0 && canvas.height > 0) {
        gl.viewport(0, 0, canvas.width, canvas.height);
      }
    }
  };

  canvas.addEventListener('webglcontextlost', onContextLost, false);
  canvas.addEventListener('webglcontextrestored', onContextRestored, false);

  buildResources();

  function resize(w, h, scale = 1) {
    if (destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (typeof w !== 'number' || !Number.isFinite(w) || w <= 0) {
      throw new Error('resize width must be a finite positive number');
    }
    if (typeof h !== 'number' || !Number.isFinite(h) || h <= 0) {
      throw new Error('resize height must be a finite positive number');
    }
    if (typeof scale !== 'number' || !Number.isFinite(scale) || scale <= 0 || scale > 4) {
      throw new Error('resize dpr must be a finite positive number at most 4');
    }
    width = w;
    height = h;
    dpr = scale;
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    if (!contextLost) {
      gl.viewport(0, 0, canvas.width, canvas.height);
    }
  }

  function render(state, opts = {}) {
    if (destroyed) {
      throw new Error('renderer is destroyed');
    }
    if (contextLost || !gl || !program) {
      return;
    }
    if (typeof state !== 'object' || state === null || typeof state.config !== 'object') {
      throw new Error('render requires a game state');
    }
    const highContrast =
      opts.highContrast === true || options.highContrast === true;
    const palette = highContrast ? HIGH_CONTRAST_PALETTE : PALETTE;
    const config = state.config;
    const gw = config.width;
    const gh = config.height;
    const scale = Math.min(width / gw, height / gh);
    const ox = (width - gw * scale) / 2;
    const oy = (height - gh * scale) / 2;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.05, 0.08, 0.12, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const skyVerts = [];
    pushRect(
      skyVerts,
      -ox / scale,
      -oy / scale,
      (width - ox) / scale,
      (height - oy) / scale,
      palette.sky
    );
    const groundVerts = [];
    pushRect(
      groundVerts,
      0,
      gh - config.groundHeight,
      gw,
      gh,
      palette.ground
    );
    if (config.groundHeight > 0) {
      pushRect(
        groundVerts,
        0,
        gh - config.groundHeight,
        gw,
        gh - config.groundHeight + Math.min(12, config.groundHeight),
        palette.grass
      );
    }
    const sceneryVerts = [];
    pushCircle(sceneryVerts, gw - 70, 64, 42, palette.sun);
    pushCircle(sceneryVerts, 96, 110, 26, palette.cloud);
    pushCircle(sceneryVerts, 132, 102, 34, palette.cloud);
    pushCircle(sceneryVerts, 320, 180, 22, palette.cloud);
    pushCircle(sceneryVerts, 352, 172, 30, palette.cloud);
    const pipeVerts = [];
    for (const pipe of state.pipes) {
      pushRect(pipeVerts, pipe.x, 0, pipe.x + config.pipeWidth, pipe.gapY, palette.pipe);
      pushRect(
        pipeVerts,
        pipe.x,
        pipe.gapY + config.gapHeight,
        pipe.x + config.pipeWidth,
        gh - config.groundHeight,
        palette.pipe
      );
    }
    const birdVerts = [];
    pushCircle(
      birdVerts,
      state.bird.x,
      state.bird.y,
      config.birdRadius,
      palette.bird
    );
    pushCircle(
      birdVerts,
      state.bird.x + config.birdRadius * 0.35,
      state.bird.y + config.birdRadius * 0.3,
      config.birdRadius * 0.35,
      palette.eye
    );

    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.uniform1f(uScaleLoc, scale);
    gl.uniform2f(uOffsetLoc, ox, oy);

    function uploadAndDraw(arr) {
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, arr.length / 6);
    }
    uploadAndDraw(skyVerts);
    uploadAndDraw(groundVerts);
    uploadAndDraw(sceneryVerts);
    uploadAndDraw(pipeVerts);
    uploadAndDraw(birdVerts);

    if (state.phase === 'paused' || state.phase === 'gameover') {
      const overlay = state.phase === 'paused'
        ? palette.pausedOverlay
        : palette.gameoverOverlay;
      const overlayVerts = [];
      pushRect(overlayVerts, 0, 0, gw, gh, overlay);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      uploadAndDraw(overlayVerts);
      gl.disable(gl.BLEND);
    }

    gl.bindVertexArray(null);
  }

  function destroy() {
    if (destroyed) {
      return;
    }
    canvas.removeEventListener('webglcontextlost', onContextLost, false);
    canvas.removeEventListener('webglcontextrestored', onContextRestored, false);
    deleteResources();
    destroyed = true;
  }

  function info() {
    return {
      backend: 'webgl2',
      width,
      height,
      dpr,
      contextLost,
      destroyed
    };
  }

  return {resize, render, destroy, info};
}
