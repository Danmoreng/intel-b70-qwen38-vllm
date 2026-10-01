import test from 'node:test';
import assert from 'node:assert/strict';

test('app module imports in Node with no DOM side effects and rejects non-element roots', async () => {
  await import('../src/config.js');
  await import('../src/random.js');
  await import('../src/input.js');
  await import('../src/simulation.js');
  await import('../src/renderer.js');
  const app = await import('../src/app.js');
  assert.equal(typeof app.mountApp, 'function');
  assert.throws(() => app.mountApp(null), TypeError);
  assert.throws(() => app.mountApp(42), TypeError);
});
