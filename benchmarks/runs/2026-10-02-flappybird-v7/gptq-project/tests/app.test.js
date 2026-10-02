import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as app from '../src/app.js';

test('app module imports safely in Node and exposes the API', () => {
  assert.equal(typeof app.mountApp, 'function');
  assert.throws(() => app.mountApp(null), TypeError);
  assert.throws(() => app.mountApp({}), TypeError);
});
