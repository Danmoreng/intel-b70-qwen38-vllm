"""One bounded eager diagnostic to expose kernels hidden by FULL graph replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from profile_capture import ProfileWorkerExtension, install_event_capture


class EagerTraceExtension(ProfileWorkerExtension):
    def install_eager_trace(self):
        runner = self.model_runner
        assert runner.vllm_config.model_config.enforce_eager
        assert str(runner.vllm_config.compilation_config.cudagraph_mode).endswith('NONE')
        identity = install_event_capture(runner)
        identity['graphs_already_captured'] = False
        identity['scope'] = 'Eager diagnostic; not the serving graph path or a throughput comparison'
        return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--engine-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    raw = gzip.decompress(args.panel.read_bytes()); panel = json.loads(raw)
    window = next(w for w in panel['windows'] if w['name'] == 'code-102752')
    assert len(window['ids']) == 102752
    config = json.loads(args.engine_config.read_text())
    assert config['enforce_eager'] and config['compilation_config']['cudagraph_mode'] == 'NONE'
    assert config['speculative_config'] == {'method': 'mtp', 'num_speculative_tokens': 3}
    assert not {'model', 'quantization', 'worker_extension_cls'}.intersection(config)
    args.out.mkdir()
    llm = LLM(model=args.model, quantization='exl3', **config,
              worker_extension_cls='run_eager_kernel_trace.EagerTraceExtension')
    short = next(w for w in panel['windows'] if w['name'] == 'code-4096')
    llm.generate([{'prompt_token_ids': short['ids'], 'cache_salt': 'eager-trace-warmup'}],
                 SamplingParams(temperature=0, max_tokens=32, ignore_eos=True), use_tqdm=False)
    inventory = llm.collective_rpc('vocabulary_inventory')
    assert all(h['full_rows'] == 248320 and h['pruned_rows'] == 65536 for i in inventory for h in i['heads'])
    identity = llm.collective_rpc('install_eager_trace')
    llm.collective_rpc('begin_kernel_trace', args=(8,))
    result = llm.generate([{'prompt_token_ids': window['ids'], 'cache_salt': 'bounded-eager-code-103k'}],
                         SamplingParams(temperature=0, top_p=1, top_k=-1, seed=20261001,
                                        max_tokens=128, ignore_eos=True), use_tqdm=False)
    assert len(result) == 1 and result[0].prompt_token_ids == window['ids']
    assert len(result[0].outputs) == 1 and len(result[0].outputs[0].token_ids) == 128
    path = args.out / 'kernel-trace.json'
    capture = llm.collective_rpc('finish_kernel_trace', args=(str(path),))
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    assert meta['pure_decode_cycles'] == meta['runner_cycles'] == 8
    assert all(b['requests'] == 1 and b['actual_rows'] == 4 and
               b['target_graph_mode'].endswith('NONE') for b in meta['batches'])
    summary = {'status': 'COMPLETE', 'kind': 'BOUNDED_EAGER_KERNEL_TRACE_NOT_SERVING_BENCHMARK',
               'engine_config': config, 'panel_sha256': hashlib.sha256(raw).hexdigest(),
               'inventory': inventory, 'capture_identity': identity, 'trace': capture,
               'output_ids_sha256': hashlib.sha256(json.dumps(result[0].outputs[0].token_ids,
                                                              separators=(',', ':')).encode()).hexdigest(),
               'scope': 'One103K C1 request,128 outputs,8 traced pure-decode cycles. Eager mode changes dispatch/graph pools. Use only to identify kernel work; require serving-graph evidence for end-to-end benefit.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
