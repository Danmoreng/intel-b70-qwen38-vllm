import test from 'node:test';
import assert from 'node:assert/strict';
import {createRenderer} from '../src/renderer.js';

test('createRenderer requires an actual canvas', () => {
  assert.throws(() => createRenderer(null), TypeError);
  assert.throws(() => createRenderer({}), TypeError);
});

test('createRenderer throws a clear error when WebGL2 is unavailable', () => {
  const fakeCanvas = {getContext: () => null};
  assert.throws(() => createRenderer(fakeCanvas), /WebGL2 is not available/);
});
