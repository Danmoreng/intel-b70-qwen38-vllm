// Public API contract: specs/scores.md
// A small pure persistence adapter: storage is injected, no DOM at import.
// Every list/record/clear performs real storage accesses, so access
// failures are always reported; stored content is only ever JSON-parsed.

const SAFE_MAX = Number.MAX_SAFE_INTEGER;

function isSafeInt(v) {
  return Number.isInteger(v) && v >= 0 && v <= SAFE_MAX;
}

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function validEntry(e) {
  if (!isPlainObject(e)) {
    return false;
  }
  const keys = Object.keys(e).sort();
  if (keys.length !== 4 || keys[0] !== 'frames' || keys[1] !== 'name' || keys[2] !== 'score' || keys[3] !== 'seed') {
    return false;
  }
  if (typeof e.name !== 'string' || e.name.trim() === '' || e.name.length > 24) {
    return false;
  }
  if (!isSafeInt(e.score) || !isSafeInt(e.frames)) {
    return false;
  }
  return Number.isInteger(e.seed) && e.seed >= 0 && e.seed <= 0xffffffff;
}

function copyEntry(e) {
  return {name: e.name.trim(), score: e.score, frames: e.frames, seed: e.seed};
}

export function createScoreStore(storage, key = 'flappybird.scores.v1') {
  if (!isPlainObject(storage) ||
      typeof storage.getItem !== 'function' ||
      typeof storage.setItem !== 'function' ||
      typeof storage.removeItem !== 'function') {
    throw new TypeError('storage must provide getItem, setItem and removeItem');
  }

  function persist(entries) {
    storage.setItem(key, JSON.stringify({version: 1, entries}));
  }

  // Missing data, malformed JSON, unknown version or invalid entries all
  // recover to an empty list; storage access failures surface as clear errors.
  function load() {
    const raw = storage.getItem(key);
    if (raw === null || raw === undefined) {
      persist([]);
      return [];
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      data = null;
    }
    if (!isPlainObject(data) || data.version !== 1 || !Array.isArray(data.entries)) {
      persist([]);
      return [];
    }
    return data.entries.filter(validEntry).map(copyEntry);
  }

  // Fresh detached arrays on every call; nothing internal is shared out.
  function list() {
    return load();
  }

  // Rejects invalid entries without any mutation; copies valid inputs, keeps
  // the best ten by score descending then frames ascending (insertion order
  // preserved for exact ties) and returns a detached list.
  function record(entry) {
    if (!validEntry(entry)) {
      throw new RangeError('score entry is invalid');
    }
    const entries = load();
    entries.push(copyEntry(entry));
    entries.sort((a, b) => b.score - a.score || a.frames - b.frames);
    const kept = entries.slice(0, 10);
    persist(kept);
    return kept.map(copyEntry);
  }

  function clear() {
    storage.getItem(key);
    storage.removeItem(key);
  }

  return {list, record, clear};
}
