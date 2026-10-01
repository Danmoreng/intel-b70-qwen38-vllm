import test from 'node:test';
import assert from 'node:assert/strict';
import {mountApp} from '../src/app.js';

test('app module is Node-importable without DOM side effects', () => {
  assert.equal(typeof mountApp, 'function');
  assert.throws(() => mountApp(null), TypeError);
});
