# QueueKit agent coding fixture v2

This fixture replaces the unreplayable private Pi coding trajectory in the
public release benchmark. Every run copies `project/` into a new ignored
`benchmark-results/` directory. The model receives two linked tasks in one
conversation and can list, read, write and run the visible tests. The runner
executes `acceptance/test_task1.py` after the first task and
`acceptance/test_task2.py` after the second. Those tests are never copied into
the model's workspace. `manifest.json` freezes the fixture bytes.

Run on an exclusive, healthy production-policy endpoint:

```bash
python3 scripts/run-coding-benchmark.py \
  --output-root benchmark-results/production-release-v1/coding-agent-v2
```

Record the live container image ID, policy hash, fixture manifest hash, task
acceptance, request and tool counts, full-session wall time, context range,
newly computed prefill, cached prompt tokens, native decode time and MTP
acceptance. This is one adaptive run; compare future versions on the exact
same fixture and runner settings, and report variation across repeats when
available. Keep generated source and conversation logs local unless reviewed
for publication.

Version 2 states the JSON adapter restore argument explicitly as `command["snapshot"]`. Version 1 is retained unchanged for audit only.
