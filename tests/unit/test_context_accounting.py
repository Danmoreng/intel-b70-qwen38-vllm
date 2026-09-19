"""Native finished-request accounting and cached-prefill regression cases."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('context_benchmark',Path(__file__).parents[2]/'scripts/context-benchmark.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class Accounting(unittest.TestCase):
    def test_waits_for_finished_record_and_uses_computed_tokens(self):
        values={k:0.0 for k in bench.COUNTER_CANDIDATES}
        def snap(v):return {'values':v,'resolved_names':{},'accepted_per_position':{}}
        after={**values,'finished_requests':1,'computed_prompt_tokens':1024,
               'prefix_cache_hits':7168,'prefix_cache_queries':8192,
               'native_prefill_seconds':2,'native_decode_seconds':1,
               'mtp_draft_tokens':4,'mtp_accepted_tokens':2}
        events=[{'choices':[{'delta':{'content':'test'},'finish_reason':'length'}]},
                {'usage':{'prompt_tokens':8192,'completion_tokens':3},'choices':[]}]
        wire=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n'
        with tempfile.TemporaryDirectory() as d, \
             patch.object(bench,'ensure_idle'), \
             patch.object(bench,'snapshot',side_effect=[snap(values),snap(values),snap(after)]) as snapshot, \
             patch.object(bench.urllib.request,'urlopen',return_value=io.BytesIO(wire.encode())), \
             patch.object(bench.time,'sleep'):
            r=bench.request('http://localhost/v1/chat/completions','http://localhost','model',[],3,
                            str(Path(d)/'raw.jsonl'),'warm',8192)
        self.assertEqual(snapshot.call_count,3)
        self.assertEqual(r['native_prefill_tokens_per_second'],512)
        self.assertEqual(r['native_decode_tokens_per_second'],2)
        self.assertEqual(r['computed_prompt_tokens'],1024)

if __name__=='__main__':unittest.main()
