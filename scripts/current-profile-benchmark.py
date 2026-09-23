#!/usr/bin/env python3
"""Current-profile phase, concurrency, prefix-cache and context benchmark.

The script never starts, stops or restarts the model service. Without
``--execute`` it only prints the selected plan. Execution is refused unless the
live container advertises ``--max-num-seqs 4`` and is idle at preflight.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from meaningful_benchmark import (
    CHAT_TEMPLATE_KWARGS, CORPUS, SAMPLING, load_corpus, make_prompt,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = "http://127.0.0.1:8081"
MODEL = "Qwen3.8-27B"
METRIC_NAMES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_cached": "vllm:prompt_tokens_cached_total",
    "prefill_computed": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "generation_tokens": "vllm:generation_tokens_total",
    "finished_requests": "vllm:request_success_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "queue_sum": "vllm:request_queue_time_seconds_sum",
    "queue_count": "vllm:request_queue_time_seconds_count",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    group: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    repeats: int
    shared_prefix_fraction: float = 0.0
    prompt_set: str = ""
    prompt_style: str = "source-review"
    expected_peak_running: int | None = None
    warmups: int = 1


def http_json(base: str, path: str, data: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    request = urllib.request.Request(
        base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    sample_re = re.compile(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([^\s]+)$")
    label_re = re.compile(r'(\w+)="((?:\\.|[^"])*)"')
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = sample_re.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = {
            key: bytes(value_text, "utf-8").decode("unicode_escape")
            for key, value_text in label_re.findall(match.group(2) or "")
        }
        samples.append((match.group(1), labels, value))
    return samples


def metric_snapshot(base: str) -> tuple[dict[str, float], str]:
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        raw = response.read().decode()
    samples = parse_prometheus(raw)
    values: dict[str, float] = {}
    for key, metric_name in METRIC_NAMES.items():
        values[key] = sum(value for name, _, value in samples if name == metric_name)
    for reason in ("capacity", "deferred"):
        values[f"waiting_{reason}"] = sum(
            value
            for name, labels, value in samples
            if name == "vllm:num_requests_waiting_by_reason"
            and labels.get("reason") == reason
        )
    return values, raw


def arg_after(command: list[str], name: str, default: str | None = None) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return default


def inspect_container(container: str) -> dict[str, Any]:
    raw = subprocess.check_output(["docker", "inspect", container], text=True, timeout=30)
    item = json.loads(raw)[0]
    command = item["Config"]["Cmd"]
    return {
        "id": item["Id"],
        "image": item["Config"]["Image"],
        "created": item["Created"],
        "command": command,
        "max_num_seqs": int(arg_after(command, "--max-num-seqs", "-1")),
        "max_num_batched_tokens": int(arg_after(command, "--max-num-batched-tokens", "-1")),
        "max_model_len": int(arg_after(command, "--max-model-len", "-1")),
        "watermark": float(arg_after(command, "--watermark", "0.0")),
        "reserve_full_isl_explicit": "--scheduler-reserve-full-isl" in command,
    }


def load_scenarios(path: Path) -> list[Scenario]:
    data = json.loads(path.read_text())
    scenarios = [Scenario(**row) for row in data]
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    for scenario in scenarios:
        if not 1 <= scenario.concurrency <= 4:
            raise ValueError(f"invalid concurrency in {scenario.name}")
        if not 0.0 <= scenario.shared_prefix_fraction <= 1.0:
            raise ValueError(f"invalid shared_prefix_fraction in {scenario.name}")
        if scenario.expected_peak_running is not None and not (
            1 <= scenario.expected_peak_running <= scenario.concurrency
        ):
            raise ValueError(f"invalid expected_peak_running in {scenario.name}")
        if scenario.prompt_style != "source-review":
            raise ValueError(f"obsolete prompt_style in {scenario.name}; use source-review")
        if scenario.warmups < 0:
            raise ValueError(f"invalid warmup count in {scenario.name}")
    return scenarios


def token_count(base: str, prompt: str) -> int:
    result = http_json(base, "/tokenize", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }, timeout=120)
    return int(result["count"])


def stream_completion(
    base: str,
    payload: dict[str, Any],
    marker: str,
    start_barrier: threading.Barrier,
    stream_path: Path,
) -> dict[str, Any]:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start_barrier.wait()
    start = time.monotonic()
    first: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    output_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    token_event_times: list[float] = []
    events = 0
    try:
        with urllib.request.urlopen(request, timeout=7200) as response, stream_path.open("w") as stream:
            for raw_line in response:
                if not raw_line.startswith(b"data: ") or raw_line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(raw_line[6:])
                elapsed = time.monotonic() - start
                stream.write(json.dumps({"elapsed_s": elapsed, "event": event}) + "\n")
                events += 1
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content = delta.get("content") or ""
                    text = reasoning + content
                    if text and first is None:
                        first = elapsed
                    if text:
                        token_event_times.append(elapsed)
                    output_parts.append(text)
                    reasoning_parts.append(reasoning)
                    content_parts.append(content)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    wall = time.monotonic() - start
    output = "".join(output_parts)
    stream_path.with_name(stream_path.name.replace("stream-", "response-").replace(".jsonl", ".txt")).write_text(
        "REASONING\n" + "".join(reasoning_parts) + "\nCONTENT\n" + "".join(content_parts)
    )
    stream_gaps = [
        later - earlier for earlier, later in zip(token_event_times, token_event_times[1:])
    ]
    return {
        "marker": marker,
        "wall_s": wall,
        "ttft_s": first,
        "post_first_generation_s": None if first is None else max(0.0, wall - first),
        "usage": usage,
        "finish_reason": finish_reason,
        "events": events,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "content_chars": len("".join(content_parts)),
        "reasoning_chars": len("".join(reasoning_parts)),
        "max_stream_gap_s": max(stream_gaps, default=0.0),
        "stream_gaps_over_2s": sum(gap > 2.0 for gap in stream_gaps),
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def read_card_energy_uj() -> int | None:
    for label_path in sorted(Path("/sys/class/hwmon").glob("hwmon*/energy*_label")):
        try:
            if label_path.read_text().strip() != "card":
                continue
            input_path = label_path.with_name(label_path.name.replace("_label", "_input"))
            return int(input_path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def wait_idle(base: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metrics, _ = metric_snapshot(base)
        if metrics["running"] == 0 and metrics["waiting"] == 0:
            return
        time.sleep(0.5)
    raise TimeoutError("engine did not become idle")


def delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def run_once(
    base: str,
    container: str,
    scenario: Scenario,
    repeat: int,
    run_dir: Path,
    sources: list[dict[str, str]],
    namespace: str,
) -> dict[str, Any]:
    wait_idle(base)
    case_dir = run_dir / f"{scenario.name}-r{repeat + 1}"
    case_dir.mkdir(parents=True)
    prompts: list[tuple[str, int, str, list[str]]] = []
    for index in range(scenario.concurrency):
        if scenario.shared_prefix_fraction == 1.0:
            marker = f"B70-{namespace}-{scenario.name}-Q{index + 1}"
        elif scenario.prompt_set:
            logical_index = repeat * scenario.concurrency + index + 1
            marker = f"B70-{namespace}-{scenario.prompt_set}-Q{logical_index}"
        else:
            marker = f"B70-{namespace}-{scenario.name}-R{repeat + 1}-Q{index + 1}"
        prompt, actual, included = make_prompt(
            sources, scenario.prompt_tokens, marker,
            lambda content: token_count(base, content),
            scenario.shared_prefix_fraction,
            f"{namespace}-{scenario.name}" if scenario.shared_prefix_fraction == 1.0 else namespace,
        )
        prompts.append((prompt, actual, marker, included))

    before, before_raw = metric_snapshot(base)
    energy_before_uj = read_card_energy_uj()
    docker_log_since = str(int(time.time()) - 1)
    (case_dir / "metrics-before.prom").write_text(before_raw)
    stop_monitor = threading.Event()
    samples: list[dict[str, Any]] = []
    monitor_start = time.monotonic()

    def monitor() -> None:
        while not stop_monitor.is_set():
            try:
                values, _ = metric_snapshot(base)
                samples.append({"elapsed_s": time.monotonic() - monitor_start, **values})
            except Exception as error:  # Preserve monitor failures without killing requests.
                samples.append({"elapsed_s": time.monotonic() - monitor_start, "error": repr(error)})
            stop_monitor.wait(0.25)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    barrier = threading.Barrier(scenario.concurrency + 1)
    batch_start = time.monotonic()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as executor:
            for index, (prompt, actual, marker, included) in enumerate(prompts):
                (case_dir / f"prompt-{index + 1}.txt").write_text(prompt)
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": scenario.output_tokens,
                    **SAMPLING,
                    "seed": 770000 + repeat * 100 + index,
                    "ignore_eos": True,
                    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                (case_dir / f"request-{index + 1}.json").write_text(
                    json.dumps(
                        {
                            **payload,
                            "messages": f"<frozen-source-prompt:{actual} tokens>",
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "marker": marker,
                            "source_paths": included,
                        },
                        indent=2,
                    )
                )
                futures.append(
                    executor.submit(
                        stream_completion,
                        base,
                        payload,
                        marker,
                        barrier,
                        case_dir / f"stream-{index + 1}.jsonl",
                    )
                )
            barrier.wait()
            results = [future.result() for future in futures]
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=5)
    batch_wall = time.monotonic() - batch_start
    wait_idle(base)
    after, after_raw = metric_snapshot(base)
    energy_after_uj = read_card_energy_uj()
    (case_dir / "metrics-after.prom").write_text(after_raw)
    with (case_dir / "metrics.jsonl").open("w") as output:
        for sample in samples:
            output.write(json.dumps(sample) + "\n")
    log_result = subprocess.run(
        ["docker", "logs", "--since", docker_log_since, container],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    server_log = log_result.stdout + log_result.stderr
    (case_dir / "server.log").write_text(server_log)
    server_preemption_lines = [
        line for line in server_log.splitlines() if "preempt" in line.lower()
    ]

    completion_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in results)
    requested_prompt_tokens = sum((row.get("usage") or {}).get("prompt_tokens", 0) for row in results)
    prompt_observed = delta(after, before, "prompt_tokens")
    prefill_computed = delta(after, before, "prefill_computed")
    prompt_cached = delta(after, before, "prompt_cached")
    prefill_seconds = delta(after, before, "prefill_seconds")
    decode_seconds = delta(after, before, "decode_seconds")
    finished_requests = delta(after, before, "finished_requests")
    drafts = delta(after, before, "draft_tokens")
    accepted = delta(after, before, "accepted_tokens")
    valid_samples = [sample for sample in samples if "error" not in sample]
    transitions: list[dict[str, Any]] = []
    previous_state: tuple[float, ...] | None = None
    for sample in valid_samples:
        state = (
            sample["running"],
            sample["waiting"],
            sample["waiting_capacity"],
            sample["preemptions"],
        )
        if state != previous_state:
            transitions.append(
                {
                    "elapsed_s": sample["elapsed_s"],
                    "running": sample["running"],
                    "waiting": sample["waiting"],
                    "waiting_capacity": sample["waiting_capacity"],
                    "kv_cache_usage": sample["kv"],
                    "preemptions_total": sample["preemptions"],
                }
            )
            previous_state = state
    post_first_tokens = max(0, completion_tokens - len(results))
    ttft_values = [row["ttft_s"] for row in results if row["ttft_s"] is not None]
    wall_values = [row["wall_s"] for row in results]
    wave_prefill_end_s = max(ttft_values, default=0.0)
    wave_decode_start_s = min(ttft_values, default=0.0)
    wave_end_s = max(wall_values, default=0.0)
    wave_decode_window_s = max(0.0, wave_end_s - wave_decode_start_s)
    full_overlap_start_s = max(ttft_values, default=0.0)
    full_overlap_end_s = min(wall_values, default=0.0)
    full_overlap_samples = [
        sample
        for sample in valid_samples
        if full_overlap_start_s <= sample["elapsed_s"] <= full_overlap_end_s
    ]
    full_overlap_sampled_s = None
    full_overlap_generation_tokens = None
    if len(full_overlap_samples) >= 2:
        full_overlap_sampled_s = (
            full_overlap_samples[-1]["elapsed_s"]
            - full_overlap_samples[0]["elapsed_s"]
        )
        full_overlap_generation_tokens = (
            full_overlap_samples[-1]["generation_tokens"]
            - full_overlap_samples[0]["generation_tokens"]
        )
    tpot_values = []
    request_decode_tps_values = []
    for row in results:
        usage = row.get("usage") or {}
        request_post_first_tokens = max(0, int(usage.get("completion_tokens", 0)) - 1)
        generation_s = row.get("post_first_generation_s")
        if request_post_first_tokens and generation_s and generation_s > 0:
            tpot_values.append(generation_s / request_post_first_tokens)
            request_decode_tps_values.append(request_post_first_tokens / generation_s)
    energy_delta_j = None
    if energy_before_uj is not None and energy_after_uj is not None and energy_after_uj >= energy_before_uj:
        energy_delta_j = (energy_after_uj - energy_before_uj) / 1_000_000
    peak_running = max((sample["running"] for sample in valid_samples), default=0)
    summary = {
        "scenario": scenario.__dict__,
        "repeat": repeat + 1,
        "batch_wall_s": batch_wall,
        "requests": results,
        "legacy_aggregate_output_tokens_per_s": completion_tokens / batch_wall if batch_wall else None,
        "request_ttft_s": distribution(ttft_values),
        "request_tpot_s": distribution(tpot_values),
        "request_post_first_decode_tps": distribution(request_decode_tps_values),
        "request_e2e_s": distribution(wall_values),
        "peak_running": peak_running,
        "peak_waiting": max((sample["waiting"] for sample in valid_samples), default=0),
        "peak_waiting_capacity": max(
            (sample["waiting_capacity"] for sample in valid_samples), default=0
        ),
        "peak_kv_cache_usage": max((sample["kv"] for sample in valid_samples), default=0),
        "preemptions": delta(after, before, "preemptions"),
        "requested_prompt_tokens": requested_prompt_tokens,
        "logical_prompt_tokens_observed": prompt_observed,
        "prefill_tokens_computed": prefill_computed,
        "prompt_tokens_cached": prompt_cached,
        "prefill_recompute_excess": max(
            0.0,
            prefill_computed + prompt_cached - requested_prompt_tokens,
        ),
        "native_prefill_seconds": prefill_seconds,
        "native_decode_seconds": decode_seconds,
        "native_prefill_compute_tokens_per_s": (
            prefill_computed / prefill_seconds if prefill_seconds > 0 else None
        ),
        "native_weighted_decode_tokens_per_s": (
            post_first_tokens / decode_seconds
            if post_first_tokens and decode_seconds > 0
            else None
        ),
        "aggregate_prefill_wave_tokens_per_s": (
            requested_prompt_tokens / wave_prefill_end_s
            if requested_prompt_tokens and wave_prefill_end_s > 0
            else None
        ),
        "aggregate_prefill_window_s": wave_prefill_end_s,
        "aggregate_decode_wave_tokens_per_s": (
            post_first_tokens / wave_decode_window_s
            if post_first_tokens and wave_decode_window_s > 0
            else None
        ),
        "aggregate_decode_window_s": wave_decode_window_s,
        "effective_decode_concurrency": (
            decode_seconds / wave_decode_window_s
            if decode_seconds > 0 and wave_decode_window_s > 0
            else None
        ),
        "fully_overlapped_decode_sampled_s": full_overlap_sampled_s,
        "fully_overlapped_generation_tokens": full_overlap_generation_tokens,
        "fully_overlapped_aggregate_decode_tokens_per_s": (
            full_overlap_generation_tokens / full_overlap_sampled_s
            if full_overlap_generation_tokens is not None
            and full_overlap_sampled_s is not None
            and full_overlap_sampled_s > 0
            else None
        ),
        "completion_tokens": completion_tokens,
        "post_first_completion_tokens": post_first_tokens,
        "finished_requests_delta": finished_requests,
        "speculative_draft_tokens": drafts,
        "speculative_accepted_tokens": accepted,
        "speculative_acceptance": accepted / drafts if drafts else None,
        "queue_time_sum_s": delta(after, before, "queue_sum"),
        "queue_observations": delta(after, before, "queue_count"),
        "card_energy_j": energy_delta_j,
        "average_card_power_w": (
            energy_delta_j / batch_wall
            if energy_delta_j is not None and batch_wall > 0
            else None
        ),
        "scheduler_transitions": transitions,
        "server_preemption_lines": server_preemption_lines,
        "admission_expectation_met": scenario.expected_peak_running is None
        or peak_running == scenario.expected_peak_running,
        "all_outputs_nonempty": all(row["content_chars"] > 0 for row in results),
        "all_prompt_counts_match": all(
            int((row.get("usage") or {}).get("prompt_tokens", 0)) == prompts[index][1]
            for index, row in enumerate(results)
        ),
        "all_completion_counts_exact": all(
            int((row.get("usage") or {}).get("completion_tokens", 0))
            == scenario.output_tokens for row in results
        ),
        "all_finish_reasons_length": all(
            row["finish_reason"] == "length" for row in results
        ),
    }
    if finished_requests != scenario.concurrency:
        raise RuntimeError(
            f"finished-request accounting mismatch: expected {scenario.concurrency}, "
            f"observed {finished_requests}"
        )
    if prompt_observed != requested_prompt_tokens:
        raise RuntimeError(
            f"prompt-token accounting mismatch: endpoint={requested_prompt_tokens}, "
            f"metric={prompt_observed}"
        )
    if not summary["all_outputs_nonempty"]:
        raise RuntimeError("one or more responses were empty")
    if not summary["all_prompt_counts_match"]:
        raise RuntimeError("endpoint prompt count differs from the calibrated prompt")
    if not summary["all_completion_counts_exact"]:
        raise RuntimeError("one or more responses did not reach the output cap")
    if not summary["all_finish_reasons_length"]:
        raise RuntimeError("one or more responses did not finish with reason=length")
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_aggregate(run_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"]["name"], []).append(row)
    summary_rows = []
    for name, items in grouped.items():
        summary_rows.append(
            {
                "name": name,
                "group": items[0]["scenario"]["group"],
                "concurrency": items[0]["scenario"]["concurrency"],
                "prompt_tokens": items[0]["scenario"]["prompt_tokens"],
                "output_tokens": items[0]["scenario"]["output_tokens"],
                "repeats": len(items),
                "batch_wall_s": distribution([item["batch_wall_s"] for item in items]),
                "request_ttft_median_s": distribution(
                    [item["request_ttft_s"]["median"] for item in items]
                ),
                "request_tpot_median_s": distribution(
                    [item["request_tpot_s"]["median"] for item in items]
                ),
                "native_prefill_compute_tokens_per_s": distribution(
                    [item["native_prefill_compute_tokens_per_s"] for item in items]
                ),
                "native_weighted_decode_tokens_per_s": distribution(
                    [item["native_weighted_decode_tokens_per_s"] for item in items]
                ),
                "aggregate_prefill_wave_tokens_per_s": distribution(
                    [item["aggregate_prefill_wave_tokens_per_s"] for item in items]
                ),
                "aggregate_decode_wave_tokens_per_s": distribution(
                    [item["aggregate_decode_wave_tokens_per_s"] for item in items]
                ),
                "weighted_aggregate_prefill_wave_tokens_per_s": (
                    sum(item["requested_prompt_tokens"] for item in items)
                    / sum(item["aggregate_prefill_window_s"] for item in items)
                ),
                "weighted_aggregate_decode_wave_tokens_per_s": (
                    sum(item["post_first_completion_tokens"] for item in items)
                    / sum(item["aggregate_decode_window_s"] for item in items)
                ),
                "weighted_native_decode_per_request_tokens_per_s": (
                    sum(item["post_first_completion_tokens"] for item in items)
                    / sum(item["native_decode_seconds"] for item in items)
                ),
                "effective_decode_concurrency": (
                    sum(item["native_decode_seconds"] for item in items)
                    / sum(item["aggregate_decode_window_s"] for item in items)
                ),
                "fully_overlapped_decode_sampled_s": sum(
                    item["fully_overlapped_decode_sampled_s"] for item in items
                    if item["fully_overlapped_decode_sampled_s"] is not None
                ),
                "fully_overlapped_generation_tokens": sum(
                    item["fully_overlapped_generation_tokens"] for item in items
                    if item["fully_overlapped_generation_tokens"] is not None
                ),
                "fully_overlapped_aggregate_decode_tokens_per_s": (
                    sum(
                        item["fully_overlapped_generation_tokens"] for item in items
                        if item["fully_overlapped_generation_tokens"] is not None
                    )
                    / sum(
                        item["fully_overlapped_decode_sampled_s"] for item in items
                        if item["fully_overlapped_decode_sampled_s"] is not None
                    )
                ),
                "speculative_acceptance": (
                    sum(item["speculative_accepted_tokens"] for item in items)
                    / sum(item["speculative_draft_tokens"] for item in items)
                    if sum(item["speculative_draft_tokens"] for item in items)
                    else None
                ),
                "card_energy_j": sum(
                    item["card_energy_j"] for item in items
                    if item["card_energy_j"] is not None
                ),
                "max_running": max(item["peak_running"] for item in items),
                "max_waiting": max(item["peak_waiting"] for item in items),
                "max_kv_cache_usage": max(item["peak_kv_cache_usage"] for item in items),
                "preemptions": sum(item["preemptions"] for item in items),
                "prefill_recompute_excess": sum(
                    item["prefill_recompute_excess"] for item in items
                ),
                "all_outputs_nonempty": all(item["all_outputs_nonempty"] for item in items),
                "all_prompt_counts_match": all(item["all_prompt_counts_match"] for item in items),
                "all_completion_counts_exact": all(
                    item["all_completion_counts_exact"] for item in items
                ),
                "all_finish_reasons_length": all(
                    item["all_finish_reasons_length"] for item in items
                ),
            }
        )
    correctness_hashes: dict[str, list[str]] = {}
    for row in rows:
        if row["scenario"].get("prompt_set"):
            for request in row["requests"]:
                correctness_hashes.setdefault(request["marker"], []).append(
                    request["output_sha256"]
                )
    comparable = {
        marker: hashes for marker, hashes in correctness_hashes.items() if len(hashes) >= 2
    }
    correctness = {
        "by_marker": correctness_hashes,
        "comparable_markers": len(comparable),
        "all_comparable_outputs_equal": bool(comparable)
        and all(len(set(hashes)) == 1 for hashes in comparable.values()),
    }
    aggregate = {
        "manifest": manifest,
        "cases": rows,
        "summary": summary_rows,
        "correctness": correctness,
    }
    (run_dir / "results.json").write_text(json.dumps(aggregate, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually send inference requests")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--container", default="qwen38-vllm-production")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=ROOT.parent / "benchmarks" / "current-profile-scenarios.json",
    )
    parser.add_argument("--only", action="append", default=[], help="scenario name or group")
    parser.add_argument("--prompt-namespace", help="Pin this value across A/B arms for identical prompts; defaults to a fresh run ID")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT.parent / "benchmark-results" / "current-profile",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.only:
        selected = set(args.only)
        scenarios = [row for row in scenarios if row.name in selected or row.group in selected]
    if not scenarios:
        raise SystemExit("no scenarios selected")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    namespace = args.prompt_namespace or timestamp
    sources, corpus_sha256 = load_corpus(CORPUS)
    plan = [scenario.__dict__ for scenario in scenarios]
    if not args.execute:
        print(json.dumps({"execute": False, "note": "dry plan only; no API calls made", "corpus_sha256": corpus_sha256, "sampling": SAMPLING, "prompt_namespace": namespace, "plan": plan}, indent=2))
        return 0

    inspect = inspect_container(args.container)
    models = http_json(args.base, "/v1/models")
    metrics, _ = metric_snapshot(args.base)
    if inspect["max_num_seqs"] != 4:
        raise SystemExit(f"refusing benchmark: live max_num_seqs={inspect['max_num_seqs']}, expected 4")
    if not inspect["reserve_full_isl_explicit"]:
        raise SystemExit("refusing benchmark: --scheduler-reserve-full-isl is not explicit")
    for scenario in scenarios:
        if scenario.prompt_tokens + scenario.output_tokens > inspect["max_model_len"]:
            raise SystemExit(f"scenario {scenario.name} exceeds the live model context limit")
    if metrics["running"] or metrics["waiting"]:
        raise SystemExit(
            f"refusing benchmark: engine is busy (running={metrics['running']}, waiting={metrics['waiting']})"
        )

    run_dir = args.output_root / f"run-{timestamp}-w{inspect['watermark']:.2f}"
    run_dir.mkdir(parents=True)
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(),
        "base": args.base,
        "container": inspect,
        "models": models,
        "corpus_sha256": corpus_sha256,
        "prompt_namespace": namespace,
        "corpus_sources": len(sources),
        "sampling": SAMPLING,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "quality_note": "Responses are saved for inspection; 1024-token fixed-length output may continue after a natural EOS and is a throughput measure, not a semantic correctness grade.",
        "scenarios": plan,
        "measurement_definitions": {
            "prefill_compute_tokens_per_s": "native computed KV tokens divided by native request prefill seconds",
            "weighted_decode_tokens_per_s": "generated tokens after the first token of each request divided by summed native request decode seconds",
            "aggregate_prefill_wave_tokens_per_s": "all prompt tokens in the wave divided by time from simultaneous release until the last request emits its first token",
            "aggregate_decode_wave_tokens_per_s": "all post-first output tokens in the wave divided by time from the earliest first token until the final request completes",
            "effective_decode_concurrency": "summed native per-request decode seconds divided by aggregate decode-window seconds; aggregate decode equals weighted per-request decode multiplied by this occupancy",
            "fully_overlapped_aggregate_decode_tokens_per_s": "native generation-token counter delta divided by sampled wall time after every request emitted its first token and before any request completed",
            "tpot_s": "client time after first generated event divided by completion tokens minus one",
            "batch_wall_s": "elapsed time from simultaneous release until every request in the wave completed",
            "e2e_policy": "reported in seconds; legacy aggregate output tokens per wall second is retained only in per-case raw data",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            for warmup in range(scenario.warmups):
                print(
                    f"WARMUP {scenario.name} {warmup + 1}/{scenario.warmups}",
                    flush=True,
                )
                run_once(
                    args.base,
                    args.container,
                    scenario,
                    -scenario.warmups + warmup,
                    run_dir / "warmups",
                    sources,
                    namespace,
                )
            for repeat in range(scenario.repeats):
                print(f"START {scenario.name} repeat {repeat + 1}/{scenario.repeats}", flush=True)
                row = run_once(args.base, args.container, scenario, repeat, run_dir, sources, namespace)
                rows.append(row)
                write_aggregate(run_dir, manifest, rows)
                print(
                    f"DONE {scenario.name}: batch={row['batch_wall_s']:.2f}s, "
                    f"prefill={row['native_prefill_compute_tokens_per_s']:.2f} tok/s, "
                    f"decode={row['native_weighted_decode_tokens_per_s']:.2f} tok/s, "
                    f"running={row['peak_running']:.0f}, waiting={row['peak_waiting']:.0f}, "
                    f"preemptions={row['preemptions']:.0f}",
                    flush=True,
                )
    finally:
        write_aggregate(run_dir, manifest, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
