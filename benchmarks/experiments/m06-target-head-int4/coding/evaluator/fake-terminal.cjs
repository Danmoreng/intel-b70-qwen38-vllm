// Exercise the real CLI with dummy terminal input; never use live credentials.
Object.defineProperty(process.stdin, 'isTTY', { value: true });
Object.defineProperty(process.stdout, 'isTTY', { value: true });
process.stdin.setRawMode = () => {};
const original = process.stdout.write.bind(process.stdout);
let prompts = 0;
process.stdout.write = function (chunk, ...args) {
  const result = original(chunk, ...args);
  if (/Passwort.*:/.test(String(chunk))) {
    prompts++;
    const value = process.env.EVAL_CANCEL === '1' ? '\x03' :
      prompts === 2 && process.env.EVAL_MISMATCH === '1' ? 'Different-pass-82\n' : 'Dummy-pass-73\n';
    setTimeout(() => process.stdin.emit('data', Buffer.from(value)), 10);
  }
  return result;
};
