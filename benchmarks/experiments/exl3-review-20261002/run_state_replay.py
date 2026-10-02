"""Revisit the fixed early prose C4 case plus a >4096-token prefill."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    raw = gzip.decompress(a.panel.read_bytes())
    panel = json.loads(raw)
    window = next(w for w in panel['windows'] if w['name'] == 'prose-4096')
    assert len(window['ids']) == 4096
    config = json.loads(a.config.read_text())
    llm = LLM(model='/exl3',quantization='exl3',dtype='float16',seed=20261001,
        worker_extension_cls='state_replay.ReplayWorkerExtension',**config)
    identity = llm.collective_rpc('install_matched_state',args=(str(a.out/'steps'),))
    waves = []
    cases=[('prose-4096',window['ids'],4,64),('prose-8192',window['ids']*2,1,8)]
    if os.environ.get('B70_REPLAY_FULL_MIXED_STATE')=='1' or os.environ.get('B70_REPLAY_EARLY_TARGETS')=='1':cases=cases[:1]
    for case,ids,c,tokens in cases:
        llm.collective_rpc('arm_matched_state',args=(case,))
        results = llm.generate([dict(prompt_token_ids=ids,cache_salt=f'matched-{case}-{i}') for i in range(c)],
            SamplingParams(temperature=0,top_p=1,top_k=-1,seed=20261001,max_tokens=tokens,ignore_eos=True),use_tqdm=False)
        assert len(results) == c and all(len(r.outputs[0].token_ids) == tokens for r in results)
        captures = llm.collective_rpc('finish_matched_state')
        waves.append(dict(case=case,prompt_tokens=len(ids),concurrency=c,
            requests=[dict(req_id=r.request_id,prompt_ids=r.prompt_token_ids,output_ids=list(r.outputs[0].token_ids)) for r in results],captures=captures))
        (a.out/'progress.json').write_text(json.dumps(waves,indent=2)+'\n')
    summary = dict(status='COMPLETE_MATCHED_STATE_DIAGNOSTIC',identity=identity,
        image_scope=os.environ.get('B70_REPLAY_IMAGE_SCOPE','Existing qualified EXL3 release; no new reference model or quantization panel'),
        panel_raw_sha256=hashlib.sha256(raw).hexdigest(),config=config,waves=waves,
        limits='Finite recreated batches; not a proof for every possible batch, history or quantization. Shadow graphs include extra native attention and are not throughput measurements.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
