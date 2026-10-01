# FlappyLab

Build a complete offline, dependency-free WebGL2 Flappy Bird laboratory from this starter. Use JavaScript ES modules. Node tests and an external software-rendered browser harness are provided by the benchmark. The game must have no third-party runtime dependencies.

Each specs/*.md file is an exact public API contract. Implement all exports described there, retaining contracts from earlier stages. Pure modules must import safely in Node and must never use a wall clock or mutate caller inputs. DOM/audio/WebGL initialization must be lazy. Tests and specs are protected; place your own regression tests in tests/*.test.js. You can edit src/**, index.html, main.js, style.css and USER_GUIDE.md.

Stages are revealed one at a time in one conversation, without compaction. Finish each stage with a brief status after using the available tools to inspect, implement and test. There is no duration or token quota; implement the actual requested behavior efficiently. Tests are not a substitute for a usable app. The final release is also checked against independent acceptance cases unavailable to the agent.

Run with a static HTTP server, e.g. python3 -m http.server 8765. Entry point index.html -> main.js -> src/app.js. No bundler, remote CDN, downloaded media or build step.
