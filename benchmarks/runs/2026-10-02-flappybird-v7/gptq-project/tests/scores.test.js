import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createScoreStore} from '../src/scores.js';

function memStorage(initial) {
  const data = new Map(initial || []);
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, String(v)),
    removeItem: (k) => data.delete(k),
  };
}

test('record validates, copies, sorts and caps at ten', () => {
  const store = createScoreStore(memStorage());
  store.record({name: '  Ada  ', score: 10, frames: 5, seed: 1});
  store.record({name: 'Bob', score: 10, frames: 5, seed: 2});
  store.record({name: 'Cal', score: 10, frames: 5, seed: 3});
  store.record({name: 'Dee', score: 99, frames: 1, seed: 4});
  store.record({name: 'Eve', score: 10, frames: 4, seed: 5});
  const list = store.list();
  assert.deepEqual(list.map((e) => e.name), ['Dee', 'Eve', 'Ada', 'Bob', 'Cal']);
  assert.equal(list[2].name, 'Ada', 'names are trimmed');
  for (let i = 0; i < 12; i += 1) {
    store.record({name: `P${i}`, score: 100 + i, frames: 0, seed: i});
  }
  assert.equal(store.list().length, 10, 'best ten kept');
  const copy = store.list();
  copy[0].name = 'Hacked';
  assert.notEqual(store.list()[0].name, 'Hacked', 'list never shares references');
  for (const bad of [
    {name: '', score: 1, frames: 1, seed: 1},
    {name: 'x'.repeat(25), score: 1, frames: 1, seed: 1},
    {name: 'Ok', score: 1.5, frames: 1, seed: 1},
    {name: 'Ok', score: 1, frames: -2, seed: 1},
    {name: 'Ok', score: 1, frames: 1, seed: 0x100000000},
    {name: 7, score: 1, frames: 1, seed: 1},
    null,
    [1],
  ]) {
    assert.throws(() => store.record(bad), Error);
  }
});

test('persistence recovers valid data and resets on bad data', () => {
  const storage = memStorage();
  const a = createScoreStore(storage);
  a.record({name: 'Ann', score: 5, frames: 20, seed: 7});
  const b = createScoreStore(storage);
  assert.deepEqual(b.list(), [{name: 'Ann', score: 5, frames: 20, seed: 7}]);
  assert.ok(b.record({name: 'Bo', score: 9, frames: 3, seed: 8}).length === 2);

  const junk = memStorage([['flappybird.scores.v1', '{not json']]);
  assert.deepEqual(createScoreStore(junk).list(), []);
  const wrongVersion = memStorage([
    ['flappybird.scores.v1', JSON.stringify({version: 2, entries: []})],
  ]);
  assert.deepEqual(createScoreStore(wrongVersion).list(), []);
  const badEntry = memStorage([
    ['flappybird.scores.v1',
      JSON.stringify({version: 1, entries: [{name: 'x', score: -1, frames: 1, seed: 1}]})],
  ]);
  assert.deepEqual(createScoreStore(badEntry).list(), []);

  const failing = {
    getItem: (k) => (memStorage()).getItem(k),
    setItem: () => {
      throw new Error('quota exceeded');
    },
    removeItem: () => {},
  };
  assert.throws(() => {
    const s = createScoreStore(failing);
    s.record({name: 'Ann', score: 1, frames: 1, seed: 1});
  }, /quota/);
});

test('clear empties and persists; new instances start clean', () => {
  const storage = memStorage();
  const a = createScoreStore(storage);
  a.record({name: 'Ann', score: 5, frames: 20, seed: 7});
  a.clear();
  assert.deepEqual(a.list(), []);
  assert.deepEqual(createScoreStore(storage).list(), [], 'cleared and persisted');
  assert.throws(() => createScoreStore({getItem: () => null}), TypeError);
});
