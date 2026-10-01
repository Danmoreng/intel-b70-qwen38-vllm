"""Short generated-path diagnostics with exact tokens, top logprobs and row metadata."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from vllm import LLM, SamplingParams
from profile_capture import batch_metadata, install_event_capture


class GenerationProbeExtension:
    def install_generation_probe(self):
        runner = self.model_runner
        identity = install_event_capture(runner)
        identity.pop('graphs_already_captured')
        identity['configured_graph_mode'] = str(runner.vllm_config.compilation_config.cudagraph_mode)
        identity['enforce_eager'] = bool(runner.vllm_config.model_config.enforce_eager)
        capture = runner._exl3_event_capture
        capture.probe_batches = []

        def metadata_only(stage, batch, descriptor, function, args, kwargs):
            if stage == 'target_body':
                capture.probe_batches.append(batch_metadata(batch, descriptor))
            return function(*args, **kwargs)

        # No GPU events, synchronization, new masks or logits modifications.
        capture.traced_call = metadata_only
        return identity

    def finish_generation_probe(self):
        capture = self.model_runner._exl3_event_capture
        rows, capture.probe_batches = capture.probe_batches, []
        return rows


def encode_output(result, prompt_ids, budget):
    assert result.prompt_token_ids == prompt_ids and len(result.outputs) == 1
    output = result.outputs[0]
    assert len(output.token_ids) == budget and output.logprobs is not None
    assert len(output.logprobs) == budget
    scores = []
    for token, entries in zip(output.token_ids, output.logprobs):
        assert token in entries and len(entries) >= 5
        ranked = sorted(({'id': int(t), 'logprob': float(lp.logprob), 'rank': int(lp.rank) if lp.rank is not None else None}
                         for t, lp in entries.items()), key=lambda item: (-item['logprob'], item['id']))
        assert all(math.isfinite(item['logprob']) for item in ranked)
        chosen = next(item for item in ranked if item['id'] == token)
        scores.append({'chosen': chosen, 'top': ranked,
                       'top1_top2_margin': ranked[0]['logprob'] - ranked[1]['logprob'],
                       'chosen_gap_from_top': ranked[0]['logprob'] - chosen['logprob']})
    ids = list(output.token_ids)
    return {'token_ids': ids, 'logprobs': scores, 'text': output.text,
            'finish_reason': output.finish_reason, 'cached_tokens': getattr(result, 'num_cached_tokens', None),
            'output_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--engine-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(); assert not args.out.exists(); args.out.mkdir()
    raw = gzip.decompress(args.panel.read_bytes()); panel = json.loads(raw)
    windows = [next(w for w in panel['windows'] if w['name'] == name)
               for name in ['code-4096', 'prose-4096']]
    assert all(len(w['ids']) == 4096 for w in windows)
    config = json.loads(args.engine_config.read_text())
    assert not {'model', 'quantization', 'worker_extension_cls'}.intersection(config)
    llm = LLM(model=args.model, quantization='exl3', **config,
              worker_extension_cls='run_generation_probe.GenerationProbeExtension')
    for c in [1, 4]:
        llm.generate([{'prompt_token_ids': windows[0]['ids'], 'cache_salt': f'probe-warmup-c{c}-r{i}'}
                      for i in range(c)], SamplingParams(temperature=0, max_tokens=32, ignore_eos=True), use_tqdm=False)
    identity = llm.collective_rpc('install_generation_probe')
    waves = []
    for window in windows:
        budget = 192 if window['domain'] == 'code' else 64
        sp = SamplingParams(temperature=0, top_p=1, top_k=-1, seed=20261001, max_tokens=budget,
                            ignore_eos=True, logprobs=5)
        for c in [1, 4]:
            for mode in ['cold', 'warm']:
                prompts = [{'prompt_token_ids': window['ids'], 'cache_salt': f'probe-{window["name"]}-c{c}-r{i}'}
                           for i in range(c)]
                results = llm.generate(prompts, sp, use_tqdm=False)
                assert len(results) == c
                row = {'window': window['name'], 'concurrency': c, 'cache_mode': mode,
                       'responses': [encode_output(r, window['ids'], budget) for r in results],
                       'batches': llm.collective_rpc('finish_generation_probe')}
                waves.append(row)
                (args.out / 'progress.json').write_text(json.dumps(waves, indent=2) + '\n')
                print(json.dumps({'window': row['window'], 'c': c, 'cache': mode,
                                  'output_hashes': [r['output_ids_sha256'] for r in row['responses']]}), flush=True)
    reference = json.loads(args.reference.read_text())['waves'] if args.reference else waves
    forced = []
    for window in windows:
        source = next(w for w in reference if (w['window'], w['concurrency'], w['cache_mode']) ==
                      (window['name'], 1, 'cold'))['responses'][0]['token_ids']
        for position in ([137] if window['domain'] == 'code' else [1, 3, 19]):
            ids = window['ids'] + source[:position]
            result = llm.generate([{'prompt_token_ids': ids, 'cache_salt': f'forced-{window["name"]}-{position}'}],
                                  SamplingParams(temperature=0, top_p=1, top_k=-1, seed=20261001,
                                                 max_tokens=1, ignore_eos=True, logprobs=5), use_tqdm=False)
            assert len(result) == 1
            forced.append({'window': window['name'], 'position': position, 'prompt_tokens': len(ids),
                           'prompt_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                           'reference_expected_token': source[position],
                           'response': encode_output(result[0], ids, 1),
                           'batches': llm.collective_rpc('finish_generation_probe')})
    summary = {'status': 'COMPLETE', 'kind': 'GENERATED_PATH_LOGPROB_DIAGNOSTIC_NOT_BENCHMARK',
               'panel_sha256': hashlib.sha256(raw).hexdigest(), 'engine_config': config,
               'capture_identity': identity, 'waves': waves, 'forced_next_token': forced,
               'reference': str(args.reference) if args.reference else 'own graph-MTP3 C1 cold outputs',
               'scope': 'Frozen4K prompts, code192/prose64 outputs; C1/C4 cold/warm. Full token IDs and top logprobs retained. Four one-token forced-prefix probes share the first graph-MTP3 history across variants. Metadata-only V2 wrappers preserve logits and sampling; no GPU event profiler. No BF16 equivalence or serving throughput claim.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
