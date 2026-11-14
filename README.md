# Qwen3.8-27B on Intel Arc Pro B70

Run [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a single **32 GB
Intel Arc Pro B70** with vLLM and an OpenAI-compatible API. This recipe uses
[the EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), custom
XPU kernels and a host RAM prefix cache. It supports chat, reasoning, tool
calls, images and video.

The supported profile is frozen and validated on the reference host. The
repository includes source, launchers and measurement receipts; the qualified
Docker image and compiled binaries are local artifacts. For a new machine,
start with [installation and build requirements](docs/INSTALLATION.md).

## Current profile

<!-- BEGIN PUBLIC PROFILE -->
| Setting | Supported value |
| --- | --- |
| GPU / power cap | 1 × Intel Arc Pro B70 (32 GB) / 180 W |
| Model | Qwen3.8-27B / EXL3 4.00 bpw |
| Runtime | vLLM 0.33.0, Torch 2.14.0+xpu |
| Context budget | 262,144 input + output tokens |
| Speculative decoding | MTP, 3 draft tokens |
| Prefix cache | FP8 GPU KV + 12 GiB host RAM |
| API | `http://127.0.0.1:8081/v1`, model `Qwen3.8-27B` |
<!-- END PUBLIC PROFILE -->

The context budget includes input, reasoning, media and output. Four parallel
requests at moderate context have been tested; sixteen admitted requests share
the same memory pool. See [limits](#limits) before sizing a workload.
[Full configuration and release identities](docs/CURRENT_CONFIGURATION.md).

## Start the service

This quick start requires the exact qualified image locally, Docker access to
`/dev/dri`, Python 3, a systemd user session, `curl` and `jq`. The launchers also
check the reference host's GPU power-control path. For another host or a source
build, follow [installation](docs/INSTALLATION.md) first.

From the checkout that will own the user service:

```bash
# Initial setup only; retain an existing .env.
cp .env.example .env
./scripts/download-model.sh
# Requires write permission to the GPU power-control file.
./scripts/set-power-limit.py
./scripts/install-user-service.sh
```

Model loading takes time. Check readiness before sending requests:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/models
```

The installer enables the inference watchdog as well as the engine service.
[Status, logs, recovery and rollback](docs/OPERATIONS.md).

## Make a request

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Write a JavaScript debounce function."}],"max_tokens":512,"reasoning_effort":"none"}'
```

By default, thinking is enabled with `reasoning_effort: "medium"`, an
8,192-token thinking budget and a 16,384-token completion cap. Requests can set
a smaller completion cap. `reasoning_effort: "none"` disables thinking unless
explicitly overridden by `chat_template_kwargs.enable_thinking`. Tool calls
use automatic tool choice and the `qwen3_xml` parser.

## Measured performance

The table summarizes the **2026-11-12** serving qualification of the current
profile at 180 W. Each request produces 1,024 tokens with thinking disabled.
Values are medians over repeated waves; K denotes the input budget in multiples
of 1,024. Actual token counts vary slightly with the frozen prompt.

<!-- BEGIN PUBLIC MEASUREMENTS -->
| Workload | Waves | First token | Decode tok/s | Wave time |
| --- | ---: | ---: | ---: | ---: |
| 4K cold, one request | 5 | 1.90 s | 62.52 | 18.27 s |
| 64K cold, one request | 3 | 38.87 s | 54.37 | 57.67 s |
| 128K cold, one request | 3 | 103.87 s | 43.18 | 127.85 s |
| 16K prefix reuse, one request | 3 | 0.33 s | 67.63 | 15.45 s |
| 64K prefix reuse, one request | 3 | 0.55 s | 56.14 | 19.96 s |
| 4K each, four requests (combined decode) | 3 | 6.37 s | 189.87 | 29.05 s |
<!-- END PUBLIC MEASUREMENTS -->

Time to first token includes client-observed waiting and prompt processing.
Decode is the server's output rate after the first token; parallel rows report
combined throughput while all requests overlap. These columns measure different
intervals and should not be added together. Warm rows reuse a preceding prompt;
they are separate cases from the cold rows.

The measured native image used the qualified cache settings and COW helper.
The packaged release installs that helper automatically; its packaging checks
are separate from the full serving matrix.

The full matrix covers **20 scenarios, 70 measured waves and 124 requests**,
with zero preemptions in the successful repeat. It measured 52.53 output tok/s
across total wave time, including prompt processing and request overhead.
Eight separate tool-agent runs solved four of four tasks on each compared
profile. This small panel supports functionality; it does not establish general
coding quality. [Measurement scope, quality and raw evidence](docs/BENCHMARKS.md).

## Limits

- **262,144 total tokens:** a sequence with 261,120 input and 1,024 output tokens
  completed in qualification. Retained reasoning and media consume this budget.
- **Shared capacity:** sixteen configured slots do not provide sixteen
  simultaneous maximum-length contexts. Large requests can wait for admission.
- **Prefix caching:** the default host RAM budget is 12 GiB. Two alternating
  maximum-length histories are not guaranteed to stay cached.
- **Media:** at most 32 images or 4 videos per prompt, within the same context
  and memory budget; the image pixel cap is 4,194,304.
- **Open stability issue:** a sporadic BCS GPU-memory fault can hang inference.
  The watchdog restarts a stalled service within its recovery limits; the
  underlying fault remains unresolved. See [recovery behavior](docs/OPERATIONS.md#watchdog).
- **Quality scope:** numerical and task checks are finite. Identical text across
  runtime versions and broad accuracy improvements are not guaranteed.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALLATION.md) | Prerequisites, available artifacts and the source-build stages |
| [Configuration](docs/CURRENT_CONFIGURATION.md) | Applied settings, pins and runtime overlays |
| [Operations](docs/OPERATIONS.md) | Service control, diagnostics, recovery and immediate rollback |
| [Benchmarks](docs/BENCHMARKS.md) | Current measurements, methodology and evidence |
| [Reproduction](docs/EXL3_BENCHMARK_REPRODUCTION.md) | Refreshing measurements and required local fixtures |
| [Archive](docs/ARCHIVE.md) | Older releases, migration plans, experiments and historical results |

To check documentation links and refreshable tables without starting the engine:

```bash
python3 scripts/check-public-docs.py
```

## Sources and acknowledgements

- [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B)
  and [turboderp EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
- [0xSero EXL3 XPU source](https://github.com/0xSero/exl3xpu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Intel B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

See [NOTICE.md](NOTICE.md), [LICENSE](LICENSE) and the preserved
[EXL3 license](engine/exl3xpu/LICENSE) for attribution and licensing.
