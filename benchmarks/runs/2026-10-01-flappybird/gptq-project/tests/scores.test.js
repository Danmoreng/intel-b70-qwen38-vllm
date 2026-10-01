import test from 'node:test';
import assert from 'node:assert/strict';
import {createScoreStore} from '../src/scores.js';

function memory() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k)
  };
}

test('record rejects invalid entries without mutation and recovers persisted data', () => {
  const storage = memory();
  const store = createScoreStore(storage);
  assert.deepEqual(store.list(), []);
  assert.throws(() => store.record({name: '', score: 1, frames: 2, seed: 1}));
  assert.throws(() => store.record({name: 'A', score: -1, frames: 0, seed: 1}));
  assert.throws(() => store.record({name: 'A', score: 1, frames: 0.5, seed: 1}));
  assert.throws(() => store.record({name: 'A', score: 1, frames: 1, seed: 0x100000000}));
  assert.throws(() => store.record({name: 'A', score: 1, frames: 1}));
  assert.throws(() => store.record({name: 'A', score: 1, frames: 1, seed: 1, extra: 2}));
  assert.deepEqual(store.list(), [], 'invalid record left the store untouched');
  storage.setItem('flappybird.scores.v1', '{not json');
  assert.deepEqual(createScoreStore(memory()).list(), []);
  const broken = createScoreStore(storage);
  assert.deepEqual(broken.list(), [], 'malformed JSON recovers to empty');
  storage.setItem('flappybird.scores.v1', JSON.stringify({version: 2, entries: []}));
  assert.deepEqual(createScoreStore(storage).list(), [], 'unknown version recovers');
  storage.setItem('flappybird.scores.v1', JSON.stringify({version: 1, entries: [{name: 'Ann', score: 5, frames: 9, seed: 7}]}));
  const recovered = createScoreStore(storage);
  assert.deepEqual(recovered.list(), [{name: 'Ann', score: 5, frames: 9, seed: 7}]);
  const boom = () => {
    throw new Error('denied');
  };
  const failing = {getItem: boom, setItem: boom, removeItem: () => {}};
  assert.throws(() => createScoreStore(failing).list(), /denied/);
  const flaky = {getItem: () => 'x', setItem: () => {}, removeItem: () => {}};
  const s2 = createScoreStore(flaky);
  s2.list();
  flaky.getItem = boom;
  assert.throws(() => s2.clear(), /denied/, 'clear also reports access failures');
});

test('best ten kept, sorted score desc then frames asc, insertion order on exact ties', () => {
  const store = createScoreStore(memory());
  store.record({name: 'B', score: 50, frames: 10, seed: 1});
  store.record({name: 'A', score: 90, frames: 50, seed: 1});
  store.record({name: 'C', score: 90, frames: 40, seed: 1});
  store.record({name: 'T1', score: 70, frames: 5, seed: 1});
  store.record({name: 'T2', score: 70, frames: 5, seed: 1});
  for (let i = 0; i < 8; i++) {
    store.record({name: `L${i}`, score: 1 + i, frames: 100 + i, seed: 1});
  }
  store.record({name: 'MID', score: 60, frames: 9, seed: 1});
  const list = store.list();
  assert.equal(list.length, 10, 'best ten survive');
  assert.deepEqual(list.map((e) => e.name), ['C', 'A', 'T1', 'T2', 'MID', 'B', 'L7', 'L6', 'L5', 'L4']);
  assert.deepEqual(list[0], {name: 'C', score: 90, frames: 40, seed: 1});
});

test('inputs are copied and the store never exposes internal references; clear empties and persists', () => {
  const storage = memory();
  const store = createScoreStore(storage);
  const src = {name: '  Ada  ', score: 12, frames: 33, seed: 9};
  const first = store.record(src);
  src.name = 'Mallory';
  assert.equal(first[0].name, 'Ada', 'names are trimmed and copied');
  const a = store.list();
  a[0].score = 999999;
  a.push({name: 'x', score: 1, frames: 1, seed: 1});
  assert.notDeepEqual(store.list(), a, 'list entries are detached copies');
  store.clear();
  assert.deepEqual(store.list(), []);
  assert.deepEqual(createScoreStore(storage).list(), [], 'clear persists across instances');
  assert.throws(() => createScoreStore({getItem: () => null}));
});
