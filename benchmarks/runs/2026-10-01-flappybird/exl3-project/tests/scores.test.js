import test from 'node:test';
import assert from 'node:assert/strict';
import {createScoreStore} from '../src/scores.js';

function memStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    map,
  };
}

test('record sorts, keeps best ten, ties and detachment', () => {
  const store = createScoreStore(memStorage());
  const r1 = store.record({name: ' A ', score: 5, frames: 90, seed: 1});
  assert.equal(r1[0].name, 'A');
  store.record({name: 'B', score: 9, frames: 40, seed: 2});
  store.record({name: 'C', score: 5, frames: 20, seed: 3}); // tie on score, fewer frames -> before A
  for (let i = 0; i < 20; i++) store.record({name: `X${i}`, score: 100 + i, frames: 1, seed: 7});
  const list = store.list();
  assert.equal(list.length, 10);
  assert.equal(list[0].score, 119);
  assert.equal(list[1].score, 118);
  for (let i = 0; i < list.length - 1; i++) {
    assert.ok(list[i].score > list[i + 1].score ||
      (list[i].score === list[i + 1].score && list[i].frames <= list[i + 1].frames));
  }
  list[0].name = 'MUTATED';
  list.push({name: 'GHOST', score: 0, frames: 0, seed: 0});
  assert.equal(store.list().length, 10);
  assert.notEqual(store.list()[0].name, 'MUTATED');

  // exact tie keeps insertion order
  const t = createScoreStore(memStorage());
  t.record({name: 'First', score: 7, frames: 5, seed: 1});
  t.record({name: 'Second', score: 7, frames: 5, seed: 2});
  assert.deepEqual(t.list().map(e => e.name), ['First', 'Second']);
});

test('invalid entries rejected without mutation; recovery and malformed reset', () => {
  const store = createScoreStore(memStorage());
  store.record({name: 'A', score: 3, frames: 10, seed: 1});
  const before = JSON.stringify(store.list());
  for (const bad of [
    {name: '', score: 1, frames: 1, seed: 1},
    {name: 'x'.repeat(25), score: 1, frames: 1, seed: 1},
    {name: 'A', score: -1, frames: 1, seed: 1},
    {name: 'A', score: 1.5, frames: 1, seed: 1},
    {name: 'A', score: 1, frames: 1, seed: 4294967296},
    {name: 'A', score: 1, frames: 1},
  ]) {
    assert.throws(() => store.record(bad));
  }
  assert.equal(JSON.stringify(store.list()), before);
  assert.throws(() => createScoreStore({getItem: () => 'x'}));

  const broken = memStorage({'flappybird.scores.v1': '{"version":1,"entries":[{"name":2}]}'});
  assert.deepEqual(createScoreStore(broken).list(), []);
  const wrongVersion = memStorage({'flappybird.scores.v1': '{"version":2,"entries":[]}'});
  assert.deepEqual(createScoreStore(wrongVersion).list(), []);

  const fresh = memStorage({'flappybird.scores.v1': JSON.stringify({version: 1, entries: [{name: 'R', score: 4, frames: 2, seed: 9}]})});
  assert.deepEqual(createScoreStore(fresh).list(), [{name: 'R', score: 4, frames: 2, seed: 9}]);
});

test('clear empties and persists; storage failures throw', () => {
  const storage = memStorage();
  const store = createScoreStore(storage);
  store.record({name: 'A', score: 1, frames: 1, seed: 1});
  assert.deepEqual(store.clear(), []);
  assert.deepEqual(createScoreStore(storage).list(), []);

  const failing = {
    getItem: () => { throw new Error('quota'); },
    setItem: () => {},
    removeItem: () => {},
  };
  assert.throws(() => createScoreStore(failing), /score storage read failed/);
  const dying = memStorage();
  const s2 = createScoreStore(dying);
  dying.getItem = () => { throw new Error('boom'); };
  dying.setItem = () => { throw new Error('quota'); };
  dying.removeItem = () => { throw new Error('gone'); };
  assert.throws(() => s2.record({name: 'A', score: 1, frames: 1, seed: 1}), /score storage write failed/);
  assert.throws(() => s2.clear(), /score storage clear failed/);
});
