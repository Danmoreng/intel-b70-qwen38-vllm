"""Exercise streaming and wave accounting against a local deterministic HTTP API."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('mtp_study',Path(__file__).with_name('run-exl3-mtp-study.py'))
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        with self.server.lock:
            rows=[f'{name} {self.server.stats.get(key,0)}' for key,name in M.R.METRICS.items()]
            rounds=self.server.stats['completed']
            rows.extend([f'vllm:spec_decode_num_drafts_total {rounds}',
                         f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="0",engine="0"}} {rounds}',
                         f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="1",engine="0"}} {rounds}',
                         'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="2",engine="0"} 0'])
        self.send_response(200);self.end_headers();self.wfile.write(('\n'.join(rows)+'\n').encode())
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        count=len(payload['prompt'])
        with self.server.lock:
            cached=count if payload['cache_salt'] in self.server.seen else 0
            self.server.seen.add(payload['cache_salt'])
            for key,value in {'completed':1,'prompt_tokens':count,'generation_tokens':4,'draft_tokens':3,
                              'accepted_tokens':2,'prefill_tokens':count-cached,'cached_tokens':cached,
                              'prefill_seconds':.01,'decode_seconds':.1}.items():
                self.server.stats[key]=self.server.stats.get(key,0)+value
        prompt=[99] if self.server.bad_prompt else payload['prompt']
        token_ids=[] if self.server.missing_ids else [10,11]
        events=[{'choices':[{'text':'ab','prompt_token_ids':prompt,'token_ids':token_ids}]},
                {'choices':[{'text':'cd','token_ids':[] if self.server.missing_ids else [12,13],'finish_reason':'length'}]},
                {'choices':[],'usage':{'prompt_tokens':count,'completion_tokens':4,'total_tokens':count+4}}]
        if self.server.missing_prompt_ids:events[0]['choices'][0].pop('prompt_token_ids')
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        try:
            for event in events:self.wfile.write(b'data: '+json.dumps(event).encode()+b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
        except (BrokenPipeError,ConnectionResetError):
            # Expected when the client rejects a deliberately invalid fixture.
            pass


class WaveTests(unittest.TestCase):
    def setUp(self):
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.lock=threading.Lock();self.server.seen=set();self.server.stats={'completed':0}
        self.server.bad_prompt=False;self.server.missing_ids=False;self.server.missing_prompt_ids=False
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=patch.object(M,'BASE',f'http://127.0.0.1:{self.server.server_port}');self.base.start()
        self.energy=patch.object(M.R,'energy',return_value=None);self.energy.start()
    def tearDown(self):
        self.energy.stop();self.base.stop();self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def wave(self,mode):
        return M.wave({'name':'code-2','domain':'code','ids':[20,21]},2,mode,0,3,self.root,4)
    def test_warmup_does_not_contaminate_cold_wave_and_warm_is_observed(self):
        self.wave('warmup');cold=self.wave('cold');warm=self.wave('warm')
        self.assertEqual(cold['native']['cached_tokens'],0)
        self.assertEqual(warm['native']['cached_tokens'],4)
        self.assertEqual(cold['native']['generation_tokens'],8)
        self.assertEqual(cold['speculative_rounds_sum'],2)
        self.assertEqual(cold['accepted_tokens_by_position'],{'0':2,'1':2,'2':0})
        self.assertEqual(cold['generated_tokens_per_round'],4)
        self.assertEqual(cold['native_request_weighted_decode_tps'],40)
        responses=json.loads((self.root/'code-2-c2-cold/result.json').read_text())['responses']
        self.assertEqual([r['token_ids'] for r in responses],[[10,11,12,13]]*2)
        self.assertTrue(all(r['client_token_delivery_gaps_s']['p50']==0 for r in responses))
    def test_wrong_prompt_ids_fail_instead_of_accepting_mismatched_fixture(self):
        self.server.bad_prompt=True
        with self.assertRaisesRegex(AssertionError,'API prompt IDs differ'):self.wave('cold')
    def test_ignored_return_ids_fail_instead_of_fabricating_token_timings(self):
        self.server.missing_ids=True
        with self.assertRaises(AssertionError):self.wave('cold')
    def test_missing_prompt_ids_fail_instead_of_claiming_alignment(self):
        self.server.missing_prompt_ids=True
        with self.assertRaisesRegex(AssertionError,'did not return the frozen prompt IDs'):self.wave('cold')


class CompactReuseTests(unittest.TestCase):
    def test_compact_selection_keeps_long_c1_and_drops_long_c4(self):
        windows=[{'name':f'{d}-{n}','context_tokens':n} for n in [4096,32768,49152,102752,131072]
                 for d in ['code','prose']]
        self.assertEqual(len(M.selected_cases(windows)),40)
        cases=M.selected_cases(windows,True)
        self.assertEqual(len(cases),18)
        self.assertTrue(all(w['name']=='code-102752' and c==1 for w,c,mode in cases if w['context_tokens']>49152))

    def test_reuse_requires_exact_identity_completed_wave_and_verified_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result_dir=root/'arm-0-mtp3/code-4096-c1-cold';result_dir.mkdir(parents=True)
            settings={'gpu_memory_utilization':.965}
            summary={'window':'code-4096','concurrency':1,'cache_mode':'cold','native':{'completed':1,'generation_tokens':512}}
            raw={**summary,'responses':[{'prompt_ids_verified':True,'token_ids':[10]*512,'usage':{'prompt_tokens':2}}]}
            campaign={'status':'CURTAILED_BY_USER','image_id':'image','panel_sha256':'panel',
                      'arms':[{'depth':3,'settings':{**settings,'speculative_config':{'method':'mtp','num_speculative_tokens':3}},'waves':[summary]}]}
            (root/'campaign.json').write_text(json.dumps(campaign));(result_dir/'result.json').write_text(json.dumps(raw))
            cases=[({'name':'code-4096','ids':[20,21]},1,'cold')]
            arm=M.reuse_arm(root,'image','panel',settings,cases)
            self.assertEqual(arm['status'],'COMPLETE_REUSED_SUBSET')
            self.assertEqual(len(arm['waves']),1)
            for image,panel,cfg in [('wrong','panel',settings),('image','wrong',settings),('image','panel',{'gpu_memory_utilization':.93})]:
                with self.assertRaises(AssertionError):M.reuse_arm(root,image,panel,cfg,cases)
            with self.assertRaisesRegex(AssertionError,'Missing or duplicated'):
                M.reuse_arm(root,'image','panel',settings,[({'name':'code-131072','ids':[20,21]},4,'warm')])
            raw['responses'][0]['token_ids'].pop();(result_dir/'result.json').write_text(json.dumps(raw))
            with self.assertRaises(AssertionError):M.reuse_arm(root,'image','panel',settings,cases)
            raw['responses'][0]['token_ids'].append(10);raw['responses'][0]['prompt_ids_verified']=False
            (result_dir/'result.json').write_text(json.dumps(raw))
            with self.assertRaises(AssertionError):M.reuse_arm(root,'image','panel',settings,cases)


if __name__=='__main__':unittest.main()
