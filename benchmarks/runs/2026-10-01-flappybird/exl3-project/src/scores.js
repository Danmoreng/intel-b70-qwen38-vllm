// Local highscore persistence adapter per specs/scores.md.
// Pure module: storage object is injected; no DOM access at import.

const DEFAULT_KEY = 'flappybird.scores.v1';
const MAX_ENTRIES = 10;
const MAX_UINT32 = 0xffffffff;

function isNonNegativeSafeInt(v) {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= Number.MAX_SAFE_INTEGER;
}

function validateEntry(entry) {
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
    return null;
  }
  const name = typeof entry.name === 'string' ? entry.name.trim() : '';
  if (name.length === 0 || name.length > 24) {
    return null;
  }
  if (!isNonNegativeSafeInt(entry.score) || !isNonNegativeSafeInt(entry.frames)) {
    return null;
  }
  if (typeof entry.seed !== 'number' || !Number.isInteger(entry.seed) || seedBelowZero(entry.seed) || entry.seed > MAX_UINT32) {
    return null;
  }
  return {name, score: entry.score, frames: entry.frames, seed: entry.seed};
}

function seedBelowZero(seed) {
  return seed < 0;
}

function compareEntries(a, b) {
  if (a.score !== b.score) {
    return b.score - a.score;
  }
  return a.frames - b.frames;
}

export function createScoreStore(storage, key = DEFAULT_KEY) {
  if (storage === null || typeof storage !== 'object' ||
      typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function' || typeof storage.removeItem !== 'function') {
    throw new TypeError('score store requires storage with getItem, setItem and removeItem methods');
  }

  function fail(action, err) {
    const message = err && err.message ? err.message : String(err);
    throw new Error(`score storage ${action} failed: ${message}`);
  }

  function read() {
    let raw;
    try {
      raw = storage.getItem(key);
    } catch (err) {
      fail('read', err);
    }
    if (raw === null || raw === undefined) {
      return [];
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      return [];
    }
    if (data === null || typeof data !== 'object' || Array.isArray(data) ||
        data.version !== 1 || !Array.isArray(data.entries)) {
      return [];
    }
    const entries = [];
    for (const item of data.entries) {
      const entry = validateEntry(item);
      if (entry === null) {
        return [];
      }
      entries.push(entry);
    }
    return entries;
  }

  function write(entries) {
    try {
      storage.setItem(key, JSON.stringify({version: 1, entries}));
    } catch (err) {
      fail('write', err);
    }
  }

  let entries = read();

  function bestTen(list) {
    return list.slice().sort(compareEntries).slice(0, MAX_ENTRIES);
  }

  function record(entry) {
    const validated = validateEntry(entry);
    if (validated === null) {
      throw new Error('invalid score entry: name must be a trimmed nonempty string up to 24 characters, score and frames nonnegative safe integers, seed a uint32');
    }
    entries = bestTen(entries.concat(validated));
    write(entries);
    return list();
  }

  function list() {
    return entries.map(e => ({name: e.name, score: e.score, frames: e.frames, seed: e.seed}));
  }

  function clear() {
    entries = [];
    try {
      storage.removeItem(key);
    } catch (err) {
      fail('clear', err);
    }
    return list();
  }

  return {list, record, clear};
}
