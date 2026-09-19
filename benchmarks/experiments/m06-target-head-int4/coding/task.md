
An Observatory installation has an already configured isolated runner. After
reconfiguring login credentials, its Tasks page can show “Agent execution
disabled” even though the runner is healthy. Investigate the configuration CLI
and fix credential rotation so existing non-auth configuration survives.

Define and document the behavior of initial setup, `--force`, explicit `--host`,
and `--password-only`. Preserve runner settings and tokens, release-helper
settings, custom paths, unrelated settings and comments when rotating an
existing configuration. A password-only change must change only the password
hash. Keep intentional gateway-token rotation behavior explicit and tested.
Reject malformed or ambiguous input rather than silently dropping settings.
Keep atomic writes and restrictive file permissions; failures must leave the
original file intact. Do not print existing secrets.

Add behavioral regression tests using temporary files and dummy credentials.
Test initial setup, rotation with preexisting runner/release/custom settings,
password-only changes, explicit host behavior and failure preservation.
Use the project's existing stack and checks. Review the diff and report the
actual checks run, any failures and limitations. Do not install dependencies,
access live data, change services, commit to the original repo or deploy.


Benchmark environment: prepared dependencies are mounted read-only. Use
B70_RELEASE_BUILD=1 for npm test: the repository uses this existing flag to
exclude host-only nested sandbox integration tests inside a sandbox. Run
all other relevant tests, typecheck, lint and formatting checks normally.
The CLI configuration code is explicitly in scope for editing in this isolated
copy; the restriction on operator files refers to the live installation.
For existing files, --force rotates password and gateway credentials while
preserving all other settings unless --host explicitly changes the host.
Duplicate keys and lines that are neither blank, comments nor assignments are
invalid input and must leave the original unchanged. Do not alter existing
unrelated tests to make this task pass.

Keep test concurrency within the sandbox's 128-task limit: after building the
contracts and runner, use B70_RELEASE_BUILD=1 npm run test --workspaces --
--maxWorkers=2 (one command). The default worker count can exhaust the sandbox
process limit on this many-core host. This is an execution setting, not a test
change. The evaluator uses the same worker limit.

The frozen baseline has seven pre-existing formatting warnings. Preserve unrelated files; report these separately rather than cleaning them up:
- docs/milestones/M6.0-01-REPORT.md
- docs/research/2026-09-13-GITHUB-LANDSCAPE.md
- docs/templates/MILESTONE-REPORT.md
- docs/templates/UPSTREAM-ADOPTION.md
- packages/runner/test/sandbox.integration.test.ts
- packages/runner/test/workspace-utility.integration.test.ts
- packages/server/test/runner/end-to-end.integration.test.ts
