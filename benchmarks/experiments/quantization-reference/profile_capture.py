"""Diagnostic XPU-event timing outside captured graphs; never a throughput arm.

Target body/head, verification sampling, commit and the complete draft proposal
are timed at their actual V2 runner call sites. A captured draft graph fuses its
body/head/sampler; its enclosing timing must not be mislabeled as body alone.
"""
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap
import time
import types

import torch


MODEL_RUNNER_SHA256 = "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce"
CALL_SITES = {
    "execute_model": {
        "self.cudagraph_manager.run_fullgraph": ("target_body", 1),
        "self.cudagraph_manager.run_pw_graph": ("target_body", 1),
        "self.model": ("target_body", 1),
    },
    "sample": {
        "self.model.compute_logits_local": ("target_head", 1),
        "self.model.compute_logits": ("target_head", 1),
        "self.sampler": ("verification_sampler", 1),
        "self.rejection_sampler": ("verification_sampler", 1),
    },
    "sample_tokens": {
        "self.postprocess_sampled": ("sampled_token_commit", 1),
        "self.speculator.propose": ("draft_proposal_including_head_and_sampler", 1),
    },
}


def patch_method(original, method_name):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    sites = CALL_SITES[method_name]
    seen = dict.fromkeys(sites, 0)

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            name = ast.unparse(node.func)
            if name not in sites:
                return node
            stage, _ = sites[name]
            seen[name] += 1
            descriptor = ast.Name(id="batch_desc", ctx=ast.Load()) if method_name == "execute_model" else ast.Constant(None)
            return ast.copy_location(ast.Call(func=ast.Name(id="_event_profile_call", ctx=ast.Load()),
                args=[ast.Name(id="self", ctx=ast.Load()), ast.Constant(stage),
                      ast.Name(id="input_batch", ctx=ast.Load()), descriptor, node.func, *node.args],
                keywords=node.keywords), node)

    tree = Calls().visit(tree)
    assert seen == {key: count for key, (_, count) in sites.items()}, \
        f"V2 {method_name} timing call sites changed: {seen}"
    function = inspect.unwrap(original.__func__)
    namespace = dict(function.__globals__)
    namespace["_event_profile_call"] = event_profile_call
    exec(compile(ast.fix_missing_locations(tree), function.__code__.co_filename + ":event-profile", "exec"), namespace)
    return types.MethodType(namespace[original.__name__], original.__self__), {
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "call_sites": seen}


def batch_metadata(batch, descriptor):
    # All fields below are CPU metadata. Do not copy device request state.
    return {"req_ids": list(batch.req_ids), "requests": int(batch.num_reqs),
            "actual_rows": int(batch.num_tokens), "padded_rows": int(batch.num_tokens_after_padding),
            "scheduled_rows_by_request": batch.num_scheduled_tokens.tolist(),
            "computed_prefill_by_request": batch.num_computed_prefill_tokens_np.tolist(),
            "prefill_length_by_request": batch.prefill_len_np.tolist(),
            "has_prefill": bool(batch.has_prefill), "draft_tokens": int(batch.num_draft_tokens),
            "target_graph_mode": str(descriptor.cg_mode) if descriptor is not None else None,
            "target_graph_bucket_rows": int(descriptor.num_tokens) if descriptor is not None else None}


class EventCapture:
    def __init__(self):
        self.enabled = False
        self.cycles = []
        self.current = None
        self.trace = None
        self.trace_waiting = False
        self.trace_active = False
        self.trace_complete = False
        self.trace_cycles = 0
        self.trace_batches = []

    def begin(self):
        assert self.current is None, "Overlapping runner cycles require a different profiler"
        self.current = {"index": len(self.cycles), "start": torch.xpu.Event(enable_timing=True),
                        "stages": [], "host_start": time.monotonic()}
        self.current["start"].record()

    def end(self):
        assert self.current is not None
        self.current["end"] = torch.xpu.Event(enable_timing=True)
        self.current["end"].record()
        self.current["host_end"] = time.monotonic()
        self.cycles.append(self.current)
        self.current = None

    def call(self, stage, batch, descriptor, function, args, kwargs):
        assert self.current is not None
        if stage == "target_body":
            self.current["batch"] = batch_metadata(batch, descriptor)
        entry = {"stage": stage, "start": torch.xpu.Event(enable_timing=True),
                 "end": torch.xpu.Event(enable_timing=True), "stream": str(torch.xpu.current_stream())}
        entry["start"].record()
        host = time.monotonic()
        value = function(*args, **kwargs)
        entry["host_dispatch_s"] = time.monotonic() - host
        entry["end"].record()
        self.current["stages"].append(entry)
        return value

    def finish(self, path):
        self.enabled = False
        assert self.current is None and self.cycles
        rows = []
        for cycle in self.cycles:
            cycle["end"].synchronize()
            row = {"index": cycle["index"], "batch": cycle.get("batch"),
                   "cycle_gpu_timeline_ms": cycle["start"].elapsed_time(cycle["end"]),
                   "cycle_host_dispatch_s": cycle["host_end"] - cycle["host_start"], "stages": []}
            assert math.isfinite(row["cycle_gpu_timeline_ms"]) and row["cycle_gpu_timeline_ms"] >= 0
            for item in cycle["stages"]:
                item["end"].synchronize()
                row["stages"].append({"stage": item["stage"],
                    "gpu_timeline_ms": item["start"].elapsed_time(item["end"]),
                    "host_dispatch_s": item["host_dispatch_s"], "stream": item["stream"]})
                assert math.isfinite(row["stages"][-1]["gpu_timeline_ms"]) and row["stages"][-1]["gpu_timeline_ms"] >= 0
            rows.append(row)
        result = {"schema": 1, "cycles": rows,
                  "scope": "Diagnostic event timeline spans, including dispatch gaps; not instrument-free serving throughput. GPU event results are synchronized only after generation finishes.",
                  "draft_scope": "Draft proposal includes input preparation and all captured draft body/head/sampler steps. It does not separately measure those components inside fused graphs.",
                  "missing_fine_components": ["draft body/head/sampler decomposition inside captured graphs", "other commit/state-update operations outside postprocess_sampled"],
                  "sources": self.sources}
        Path(path).write_text(json.dumps(result, indent=2) + "\n")
        self.cycles = []
        return {"path": str(path), "cycles": len(rows)}

    def begin_trace(self, cycles):
        assert not self.enabled and self.current is None and self.trace is None
        assert 1 <= cycles <= 16
        assert torch.profiler.ProfilerActivity.XPU in torch.profiler.supported_activities(), \
            "Actual XPU kernel activity tracing is unavailable"
        self.trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU], record_shapes=False, with_stack=False)
        self.trace_waiting = True
        self.trace_complete = False
        self.trace_cycles = 0
        self.trace_max_cycles = cycles
        self.trace_batches = []
        return {"waiting_for_first_pure_decode": True, "cycles": cycles}

    def traced_call(self, stage, batch, descriptor, function, args, kwargs):
        if stage == "target_body":
            if self.trace_waiting and not batch.has_prefill:
                self.trace.start()
                self.trace_active = True
                self.trace_waiting = False
            if self.trace_active:
                self.trace_batches.append(batch_metadata(batch, descriptor))
        if self.trace_active:
            with torch.profiler.record_function(f"exl3_diagnostic/cycle{self.trace_cycles}/" + stage):
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    def trace_step_complete(self):
        if self.trace_active:
            self.trace_cycles += 1
            if self.trace_cycles == self.trace_max_cycles:
                # Bounded diagnostic only. Stop may synchronize; its wave must
                # never enter uninstrumented serving/component timing tables.
                self.trace.stop()
                self.trace_active = False
                self.trace_complete = True

    def finish_trace(self, path):
        assert self.trace_complete and not self.trace_active and not self.trace_waiting
        assert self.trace_cycles == len(self.trace_batches) == self.trace_max_cycles
        self.trace.export_chrome_trace(str(path))
        pure = sum(not batch["has_prefill"] for batch in self.trace_batches)
        metadata = {"status": "COMPLETE", "runner_cycles": self.trace_cycles,
                    "pure_decode_cycles": pure, "mixed_or_prefill_cycles": self.trace_cycles - pure,
                    "batches": self.trace_batches,
                    "scope": "Bounded runner-cycle kernel trace starting at first pure decode; later mixed cycles are labeled by CPU metadata. Trace startup/stop perturb execution; no throughput claim."}
        Path(str(path) + ".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.trace = None
        return {"path": str(path), "runner_cycles": self.trace_cycles, "pure_decode_cycles": pure}


def event_profile_call(runner, stage, batch, descriptor, function, *args, **kwargs):
    profile = runner._exl3_event_capture
    if not profile.enabled:
        return profile.traced_call(stage, batch, descriptor, function, args, kwargs)
    return profile.call(stage, batch, descriptor, function, args, kwargs)


def install_event_capture(runner):
    assert type(runner).__name__ == "XPUModelRunnerV2"
    path = Path(inspect.getfile(inspect.unwrap(runner.execute_model.__func__)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_RUNNER_SHA256
    assert runner.batch_sharder is None and runner.ubatch_runner is None
    assert runner.pcp_manager is None and runner.dp_size == 1
    assert not hasattr(runner, "_exl3_event_capture"), "Profiler already installed"
    profile = EventCapture()
    profile.sources = {}
    patched = {}
    for name in CALL_SITES:
        patched[name], profile.sources[name] = patch_method(getattr(runner, name), name)
    execute, sample_tokens = patched["execute_model"], patched["sample_tokens"]

    def execute_model(*args, **kwargs):
        if not profile.enabled:
            return execute(*args, **kwargs)
        profile.begin()
        value = execute(*args, **kwargs)
        if runner.execute_model_state is None:
            profile.end()
        return value

    def sample_tokens_wrapper(*args, **kwargs):
        value = sample_tokens(*args, **kwargs)
        if profile.enabled:
            profile.end()
        profile.trace_step_complete()
        return value

    runner._exl3_event_capture = profile
    runner.execute_model = execute_model
    runner.sample = patched["sample"]
    runner.sample_tokens = sample_tokens_wrapper
    return {"installed": True, "model_runner_sha256": MODEL_RUNNER_SHA256,
            "call_sites": profile.sources, "graphs_already_captured": True}


class ProfileWorkerExtension:
    def vocabulary_inventory(self):
        runner = self.model_runner
        draft = getattr(getattr(runner, 'speculator', None), 'model', None)
        assert draft is not None, 'MTP model must be loaded'
        heads = []
        for role, model in [('target', runner.model), ('draft', draft)]:
            for name, module in model.named_modules():
                report = getattr(module, 'exl3_loader_report', None)
                if report is None or 'lm_head' not in report['members']:
                    continue
                reduced = getattr(module, 'exl3_draft', None)
                heads.append({'role': role, 'module': name, 'bits': report['bits'],
                              'full_rows': int(module.svh.numel()),
                              'pruned_rows': int(reduced['svh'].numel()) if reduced is not None else None,
                              'pruned_tensor_logical_bytes': sum(v.numel() * v.element_size()
                                  for v in reduced.values() if isinstance(v, torch.Tensor)) if reduced is not None else 0})
        assert {h['role'] for h in heads} == {'target', 'draft'}
        free, total = torch.xpu.mem_get_info()
        return {'heads': heads, 'device_free_bytes': free, 'device_total_bytes': total,
                'torch_allocated_bytes': torch.xpu.memory_allocated(),
                'torch_reserved_bytes': torch.xpu.memory_reserved(),
                'scope': 'Post-load/capture inventory outside measurement; logical tensor bytes are not unique allocator residency.'}

    def install_event_profile(self):
        return install_event_capture(self.model_runner)

    def begin_event_profile(self):
        profile = self.model_runner._exl3_event_capture
        assert not profile.enabled and not profile.cycles and profile.current is None and profile.trace is None
        profile.enabled = True
        return {"enabled": True}

    def finish_event_profile(self, path):
        return self.model_runner._exl3_event_capture.finish(path)

    def begin_kernel_trace(self, cycles=8):
        return self.model_runner._exl3_event_capture.begin_trace(cycles)

    def finish_kernel_trace(self, path):
        return self.model_runner._exl3_event_capture.finish_trace(path)
