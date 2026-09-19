# Proposed coding task: preserve Observatory configuration during credential rotation

This is a benchmark task specification, not an instruction to change the live
installation. Run only in a prepared isolated copy of the dashboard project.

## Initial task

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

## Follow-up in the same session

Make the Tasks page accurately distinguish execution disabled by configuration
from an unreachable configured runner. The current generic safety-check wording
is misleading when no runner client was constructed. Trace the backend status
through contracts and UI; present useful English and German text consistent
with each actual condition. Do not infer completed safety checks or runner health
from an open event stream. Add relevant regression tests, retain existing API
compatibility where practical, and run the final quality checks.

## External acceptance criteria

The evaluator, not the coding agent, must hold an unchanged acceptance suite.
It must verify preservation of existing dummy runner/release settings and secret
values, intended credential changes, no leaked old secrets, atomic failure
behavior, and distinct configured-disabled versus runner-unreachable states.
Existing relevant tests must continue to pass. Neither self-reported success
nor elapsed time is sufficient. The external suite still needs implementation
and a red check against the frozen starting revision before the benchmark runs.
