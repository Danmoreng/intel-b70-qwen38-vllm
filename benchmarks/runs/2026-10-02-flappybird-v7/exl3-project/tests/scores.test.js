import test from 'node:test';
import assert from 'node:assert/strict';
import {createScoreStore} from '../src/scores.js';

function memStorage(data = {}) {
  const map = new Map(Object.entries(data));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    dump: () => map
  };
}

test('record validates, sorts, trims and keeps best ten', () => {
  const store = createScoreStore(memStorage());
  for (let i = 0; i < 12; i++) {
    store.record({name: 'P' + i, score: i, frames: 100, seed: 1});
  }
  let list = store.list();
  assert.equal(list.length, 10);
  assert.equal(list[0].score, 11);
  assert.deepEqual(list.map((e) => e.score), [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]);
  const out = store.record({name: '  Zed  ', score: 50, frames: 0, seed: 0});
  assert.equal(out[0].name, 'Zed', 'name trimmed and copied');
  // ties by score: frames ascending
  const s2 = createScoreStore(memStorage());
  s2.record({name: 'a', score: 7, frames: 30, seed: 1});
  s2.record({name: 'b', score: 7, frames: 10, seed: 2});
  s2.record({name: 'c', score: 7, frames: 30, seed: 3});
  assert.deepEqual(s2.list().map((e) => e.name), ['b', 'a', 'c']);
  for (const bad of [
    {name: '', score: 1, frames: 1, seed: 1},
    {name: 'x'.repeat(25), score: 1, frames: 1, seed: 1},
    {name: 'ok', score: -1, frames: 1, seed: 1},
    {name: 'ok', score: 1, frames: 1.5, seed: 1},
    {name: 'ok', score: 1, frames: 1, seed: 2 ** 32}
  ]) {
    assert.throws(() => s2.record(bad));
  }
});

test('list is detached; clear empties and persists', () => {
  const storage = memStorage();
  const store = createScoreStore(storage);
  store.record({name: 'a', score: 5, frames: 1, seed: 1});
  const a = store.list();
  a[0].name = 'HACKED';
  a.push({name: 'x', score: 0, frames: 0, seed: 0});
  assert.equal(store.list()[0].name, 'a', 'mutations do not reach the store');
  const other = createScoreStore(storage, 'flappybird.scores.v1');
  assert.deepEqual(other.list(), [{name: 'a', score: 5, frames: 1, seed: 1}]);
  other.clear();
  assert.deepEqual(other.list(), []);
  assert.equal(storage.dump().has('flappybird.scores.v1'), false);
  assert.deepEqual(createScoreStore(storage).list(), [], 'fresh instance sees cleared storage');
});

test('malformed persisted data resets; storage failures throw', () => {
  for (const bad of ['not json', '{"version":2,"entries":[]}', '{"version":1,"entries":[{name:"a"}]}']) {
    const store = createScoreStore(memStorage({'flappybird.scores.v1': bad}));
    assert.deepEqual(store.list(), []);
  }
  const failing = {
    getItem: () => { throw new Error('boom'); },
    setItem: () => { throw new Error('boom'); },
    removeItem: () => {}
  };
  assert.throws(() => createScoreStore(failing), /storage read failed/);
  const ok = memStorage();
  const s = createScoreStore(ok);
  ok.setItem = () => { throw new Error('quota'); };
  assert.throws(() => s.record({name: 'a', score: 1, frames: 1, seed: 1}), /storage write failed/);
});
