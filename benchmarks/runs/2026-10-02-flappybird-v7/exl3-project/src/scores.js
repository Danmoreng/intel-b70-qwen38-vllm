// Public API contract: specs/scores.md

const DEFAULT_KEY = 'flappybird.scores.v1';
const MAX_ENTRIES = 10;
const MAX_NAME_LENGTH = 24;
const MAX_UINT32 = 0xFFFFFFFF;

function isSafeNonnegInt(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function isUint32(value) {
  return isSafeNonnegInt(value) && value <= MAX_UINT32;
}

function cloneEntry(entry) {
  return {
    name: String(entry.name).trim(),
    score: entry.score,
    frames: entry.frames,
    seed: entry.seed
  };
}

export function createScoreStore(storage, key = DEFAULT_KEY) {
  if (
    storage === null ||
    typeof storage !== 'object' ||
    typeof storage.getItem !== 'function' ||
    typeof storage.setItem !== 'function' ||
    typeof storage.removeItem !== 'function'
  ) {
    throw new Error('createScoreStore expects a storage with getItem/setItem/removeItem');
  }
  if (typeof key !== 'string' || key.length === 0) {
    throw new Error('createScoreStore expects a nonempty storage key');
  }

  let entries = [];

  function validateEntry(entry) {
    if (entry === null || typeof entry !== 'object') {
      return 'entry must be an object';
    }
    if (typeof entry.name !== 'string') {
      return 'entry name must be a string';
    }
    const name = entry.name.trim();
    if (name.length === 0 || name.length > MAX_NAME_LENGTH) {
      return 'entry name must trim to 1..24 characters';
    }
    if (!isSafeNonnegInt(entry.score)) {
      return 'entry score must be a nonnegative safe integer';
    }
    if (!isSafeNonnegInt(entry.frames)) {
      return 'entry frames must be a nonnegative safe integer';
    }
    if (!isUint32(entry.seed)) {
      return 'entry seed must be a uint32';
    }
    return null;
  }

  function load() {
    let raw = null;
    try {
      raw = storage.getItem(key);
    } catch (err) {
      throw new Error('score storage read failed: ' + err.message);
    }
    if (raw === null || raw === undefined) {
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      return; // malformed JSON: reset to empty
    }
    if (
      parsed === null ||
      typeof parsed !== 'object' ||
      Array.isArray(parsed) ||
      parsed.version !== 1 ||
      !Array.isArray(parsed.entries)
    ) {
      return;
    }
    const valid = [];
    for (const entry of parsed.entries) {
      if (validateEntry(entry) !== null) {
        return; // any invalid entry: reset to empty
      }
      valid.push(cloneEntry(entry));
    }
    entries = valid;
  }

  function persist() {
    try {
      storage.setItem(
        key,
        JSON.stringify({version: 1, entries: entries.map(cloneEntry)})
      );
    } catch (err) {
      throw new Error('score storage write failed: ' + err.message);
    }
  }

  load();

  function list() {
    return entries.map(cloneEntry);
  }

  function record(entry) {
    const problem = validateEntry(entry);
    if (problem !== null) {
      throw new Error('record: ' + problem);
    }
    const copy = cloneEntry(entry);
    // Insertion-order stable for exact ties (newer entries sort after older).
    entries.push(copy);
    entries.sort((a, b) => (b.score - a.score) || (a.frames - b.frames));
    entries = entries.slice(0, MAX_ENTRIES);
    persist();
    return list();
  }

  function clear() {
    entries = [];
    try {
      storage.removeItem(key);
    } catch (err) {
      throw new Error('score storage clear failed: ' + err.message);
    }
    return list();
  }

  return {list, record, clear};
}
