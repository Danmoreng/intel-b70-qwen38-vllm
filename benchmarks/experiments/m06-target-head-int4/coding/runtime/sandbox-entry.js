process.env.B70_READONLY_DEPS = '1';
import { spawn } from 'node:child_process';
import { createConnection, createServer } from 'node:net';
const RELAY_SOCKET = '/inference/relay.sock';
const LISTEN_HOST = '127.0.0.1';
const LISTEN_PORT = 8081;
const SAFE_AGENT_PATH = /^\/agent-runtime\/[A-Za-z0-9@._+/-]+$/;
function fail(message) {
    process.stderr.write(`[b70-sandbox] ${message}\n`);
    process.exit(70);
}
const executable = process.env.B70_AGENT_EXECUTABLE;
const token = process.env.B70_INFERENCE_TOKEN;
if (executable === undefined || !SAFE_AGENT_PATH.test(executable) || executable.includes('..')) {
    fail('invalid agent executable');
}
if (token === undefined || !/^[a-f0-9]{64}$/.test(token)) {
    fail('missing inference capability');
}
const sockets = new Set();
const relay = createServer((client) => {
    if (sockets.size >= 16) {
        client.destroy();
        return;
    }
    const upstream = createConnection(RELAY_SOCKET);
    sockets.add(client);
    sockets.add(upstream);
    const forget = (socket) => () => {
        sockets.delete(socket);
    };
    client.once('close', forget(client));
    upstream.once('close', forget(upstream));
    client.once('error', () => upstream.destroy());
    upstream.once('error', () => client.destroy());
    client.pipe(upstream);
    upstream.pipe(client);
});
relay.maxConnections = 16;
relay.once('error', (error) => fail(`inference bridge failed: ${error.message}`));
relay.listen(LISTEN_PORT, LISTEN_HOST, () => {
    const child = spawn('/usr/bin/node', [executable, ...process.argv.slice(2)], {
        cwd: '/workspace',
        env: process.env,
        stdio: 'inherit',
    });
    const finish = (code, signal) => {
        relay.close();
        for (const socket of sockets)
            socket.destroy();
        process.exit(signal === null ? (code ?? 1) : 1);
    };
    child.once('error', (error) => fail(`agent launch failed: ${error.message}`));
    child.once('exit', finish);
    for (const signal of ['SIGINT', 'SIGTERM']) {
        process.once(signal, () => child.kill(signal));
    }
});
