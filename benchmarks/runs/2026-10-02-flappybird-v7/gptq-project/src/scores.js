// Local highscore persistence adapter, per specs/scores.md.
// Pure module: storage is injected; no global DOM access at import time,
// no timestamps, no random IDs, stored content is parsed but never executed.

const KEY = 'flappybird.scores.v1';
const MAX_ENTRIES = 10;
const MAX_NAME = 24;
const MAX_UINT32 = 0xffffffff;

function isValidEntry(entry) {
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
    return false;
  }
  const name = typeof entry.name === 'string' ? entry.name.trim() : '';
  if (name === '' || name.length > MAX_NAME) {
    return false;
  }
  if (typeof entry.score !== 'number' || !Number.isSafeInteger(entry.score) || entry.score < 0) {
    return false;
  }
  if (typeof entry.frames !== 'number' || !Number.isSafeInteger(entry.frames) || entry.frames < 0) {
    return false;
  }
  if (typeof entry.seed !== 'number' || !Number.isInteger(entry.seed) ||
    entry.seed < 0 || entry.seed > MAX_UINT32) {
    return false;
  }
  return true;
}

function copyEntry(entry) {
  return {
    name: entry.name.trim(),
    score: entry.score,
    frames: entry.frames,
    seed: entry.seed,
  };
}

export function createScoreStore(storage, key = KEY) {
  if (storage === null || typeof storage !== 'object' || Array.isArray(storage)) {
    throw new TypeError('createScoreStore requires a storage object');
  }
  for (const method of ['getItem', 'setItem', 'removeItem']) {
    if (typeof storage[method] !== 'function') {
      throw new TypeError(`storage.${method} must be a function`);
    }
  }

  let entries = load();

  function load() {
    const text = storage.getItem(key);
    if (text === null || text === undefined) {
      return [];
    }
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return [];
    }
    if (data === null || typeof data !== 'object' || Array.isArray(data) ||
      data.version !== 1 || !Array.isArray(data.entries)) {
      return [];
    }
    const loaded = [];
    for (const entry of data.entries) {
      if (!isValidEntry(entry)) {
        return [];
      }
      loaded.push(copyEntry(entry));
    }
    return loaded;
  }

  function persist() {
    storage.setItem(key, JSON.stringify({version: 1, entries}));
  }

  function compare(a, b) {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.frames - b.frames;
  }

  function list() {
    return entries.map(copyEntry);
  }

  function record(entry) {
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new TypeError('record requires an entry object');
    }
    if (!isValidEntry(entry)) {
      throw new RangeError('record entry failed validation');
    }
    entries.push(copyEntry(entry));
    entries.sort(compare);
    entries = entries.slice(0, MAX_ENTRIES);
    persist();
    return list();
  }

  function clear() {
    entries = [];
    persist();
    return list();
  }

  return {list, record, clear};
}
