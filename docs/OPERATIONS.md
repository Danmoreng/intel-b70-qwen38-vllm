# Service operations

[Back to README](../README.md) · [Configuration](CURRENT_CONFIGURATION.md)

Run these commands from the checkout installed into the user service. The
engine service is `b70-qwen38-vllm.service`; its container is
`b70-qwen38-vllm`. Defaults are loopback port 8081. The recovery helper targets
that endpoint and standard container name.

## Status and recovery

```bash
systemctl --user status --no-pager b70-qwen38-vllm.service
journalctl --user -u b70-qwen38-vllm.service -n 60 --no-pager
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/models
python3 scripts/run-server-exl3.py --check-only
python3 scripts/restore-production.py
```

The restore helper validates release receipts, the service checkout and the
launcher preflight before starting the service. It waits for health and checks
the running image/policy. The service uses `Type=exec`: an active systemd
process alone does not mean model loading has finished. Confirm a real reply
with the [API example](../README.md#make-a-request).

Direct launcher preflight does not load `.env`; the service and restore helper
do. If checking non-default host settings directly, load the trusted `.env`
into the shell first. `--check-only` runs a verification container but does not
load the inference model or start the service.

```bash
systemctl --user stop b70-qwen38-vllm.service
systemctl --user start b70-qwen38-vllm.service
```

An explicit stop leaves the engine stopped. The watchdog never starts an
inactive engine service. For an existing installation, rerunning
`scripts/install-user-service.sh` updates the service and watchdog units to
this checkout and enables both; it is not a read-only status command.

## Watchdog

The installer enables `b70-qwen38-inference-watchdog.timer`. Its Bash worker
requires `curl`, `jq` and GNU coreutils, with no compiler or Python dependency.
It samples HTTP health and actual prompt/output progress every 30 seconds.

| Condition | Recovery threshold |
|---|---|
| Active request with no progress | 10 minutes |
| Kernel-confirmed timed-out vLLM GPU job | 60 seconds without progress |
| API unavailable during startup | At most 15 minutes of grace |
| Continuous API outage after readiness | 10 minutes |
| Restart budget | At most three per hour, with a 15-minute cooldown |

This bounds recovery attempts; it does not guarantee recovery from every GPU
fault. The sporadic BCS GPU-memory fault remains unexplained and was also seen
with the preceding profile. Watchdog recovery does not fix its cause.

```bash
systemctl --user status --no-pager b70-qwen38-inference-watchdog.timer
journalctl --user -u b70-qwen38-inference-watchdog.service -n 60 --no-pager
bash scripts/test-inference-watchdog.sh
```

The tests use a simulated service and do not restart the real engine. State
and restart reasons live under `~/.local/state/b70-qwen38-vllm/`; events are
also in the watchdog journal. Recovery operates at any time of day. Host
shutdown schedules are independent operating-system policy.

## Immediate rollback

The rollback target is derived from the current policy and checked by the
public-documentation checker:

<!-- BEGIN IMMEDIATE ROLLBACK -->
Use the [saved exl3-v6 release](../config/releases/exl3-v6/README.md)
with `scripts/run-server-exl3-v6-rollback.sh`. Its immutable image is
`sha256:f8b0ef94cf95cfee20b0c5500db1173fdaaca4af4e3cbaa352a511e78e6a0c6d`.
<!-- END IMMEDIATE ROLLBACK -->

Validate the saved image and runtime before stopping the current engine:

```bash
./scripts/run-server-exl3-v6-rollback.sh --check-only
systemctl --user stop b70-qwen38-vllm.service
./scripts/run-server-exl3-v6-rollback.sh
```

The final command runs the rollback in the foreground with its archived policy
and RAM modules; it does not change the installed user-service target. The
engine must be stopped before another profile uses the same GPU, port or
container name. In another terminal, verify health, the model list and a real
reply. Stop the foreground rollback before restoring the current service with
`python3 scripts/restore-production.py` from the installed checkout.

Older release configurations remain under [config/releases](../config/releases).
[The archive index](ARCHIVE.md) links their own instructions, including GPTQ's
persistent user-service procedure. Do not overwrite a release manifest to
switch profiles.
