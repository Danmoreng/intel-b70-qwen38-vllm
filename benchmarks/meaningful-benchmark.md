# Meaningful serving benchmark inputs

The former context and current-profile harnesses padded prompts with repeated
`x` or `benchmark` tokens and forced fixed-length greedy output. Those results
remain historical hardware diagnostics; they are not representative coding
measurements and must not be compared directly with the revised harness.

The replacement [corpus](meaningful-corpus.json) freezes 197 public source and
documentation files from this repository. The [builder](../scripts/build-meaningful-corpus.py)
can explicitly regenerate it. Every source has a SHA-256; the overall corpus
SHA-256 is written to each new benchmark manifest. Each request selects a
deterministic source-file rotation and asks for a grounded code review, focused
tests, component explanation or technical summary. Prompts contain complete,
distinct source files. The actual token count is
recorded; the token budget is an upper bound, not an artificial exact target.
The prefix-cache scenarios resend the same complete source prompt to compare
the cold request with a warm request. Exact resend avoids mistaking a boundary
or Mamba-state eligibility effect for a failure to construct shared text.
The current runner pins the prompt namespace to `20260923-201101`. A timestamp
still gives every output directory a unique run ID. The namespace is included
in each case ID; its hash selects the source-file rotation and task wording.
Changing it therefore changes the workload, not just the result label. Keep
the namespace fixed across configurations, and use an idle or freshly started
worker to control prefix-cache state. Override `--prompt-namespace` only when
intentionally defining another fixture.
The current production benchmark uses `--fixture-root` to replay the original
full-run prompt bytes and request values for every measured request, including
both prefix scenarios. The first 64K prefix request can reuse the earlier 16K
prefix, so its cold/warm row is measured again on a fresh worker using the same
frozen 64K prompt. The previous publication used a separate 64K fixture with
namespace `20260923-212648`; its historical numbers remain in that run's summary.
Without `--fixture-root`, `--legacy-prefix-namespace` reconstructs the original
16K prompt style. Exact comparisons should use saved prompt bytes rather than
assuming prompt generation is identical across harness versions.

Short serving sweeps use the production coding sampling values: temperature
1.0, top-p 0.95, top-k 20 and a recorded per-request seed. Thinking is off so
the fixed 1,024-token response caps contain visible answers. `ignore_eos=true`
forces each request to reach that cap. The actual completion count and `length`
finish reason are checked and saved.
Prompts, streamed events and response text remain in ignored local run folders;
tracked aggregate results can omit them. The runner checks token accounting,
nonempty visible output and valid finish reasons. Semantic correctness still
requires inspecting the recorded answer or a separate task-level test, especially
because generation can continue after a natural end of the answer.

For a short profile check on the current production engine:

```bash
python3 scripts/current-profile-benchmark.py --only phase-4k-c1 --execute
```

For a short cold-cache context sweep:

```bash
./scripts/run-context-benchmark.sh benchmark-results/source-review
```

For a repeatable coding-agent workload, use the
[QueueKit fixture v2](coding-fixture/v2/README.md), which keeps the project,
tasks, tools and hidden acceptance tests fixed.
