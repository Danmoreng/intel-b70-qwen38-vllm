#!/usr/bin/env python3
"""FP16 or draft-only W4 DFlash capacity/state and frozen C1 speed screens."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE,MODEL,profile,sha
spec=importlib.util.spec_from_file_location('restore',REPO/'scripts/run-exl3-review-micro.py');restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--depth',type=int,default=3)
p.add_argument('--contexts',default='192000,96000,64000')
p.add_argument('--benchmark',action='store_true')
p.add_argument('--capacity-probe',action='store_true')
p.add_argument('--image',default='local/b70-dflash2-serving:v031')
p.add_argument('--wide-m04',action='store_true')
p.add_argument('--method',choices=['dflash','mtp'],default='dflash')
p.add_argument('--scenario-names',default='phase-4k-c1,phase-32k-c1,phase-64k-c1')
p.add_argument('--draft-sample-method',choices=['greedy','probabilistic'],default='greedy')
p.add_argument('--capture-q',action='store_true')
p.add_argument('--greedy-control',action='store_true')
p.add_argument('--draft-format',choices=['fp16','w4'],default='fp16')
a=p.parse_args()
root=REPO/'benchmark-results/dflash2-serving-v031-20261009'/a.name;root.mkdir(parents=True,exist_ok=False)
(root/'executing-controller.py').write_text(Path(__file__).read_text())
production=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())['image_id']
image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
draft=Path.home()/('.cache/exl3xpu/syvai-Qwen3.8-27B-DFlash2-W4A16-4d30ec73-fp16' if a.draft_format=='w4' else '.cache/exl3xpu/incoai-Qwen3.8-27B-DFlash2-dedf8df6-fp16')
conversion=None
if a.method=='dflash':
    conversion=json.loads((draft/'conversion.json').read_text());assert sha(draft/'model.safetensors')==(conversion['output_sha256'] if a.draft_format=='w4' else conversion['fp16_sha256']);assert sha(draft/'config.json')==conversion['config_sha256']
    assert json.loads((draft/'config.json').read_text())['dtype']=='float16'
fixture=REPO.parent/'intel-b70-qwen38-vllm/benchmark-results/meaningful-full-profile/run-20260923-201101-w0.00'
frozen=json.loads((REPO/'config/experiments/exl3-migration/full-serving-fixture-manifest.json').read_text())
for name,digest in frozen['files'].items():assert sha(fixture/name)==digest,name
state=dict(status='RUNNING',image=image,production_image=production,draft=conversion,method=a.method,depth=a.depth,draft_sample_method=a.draft_sample_method,draft_format=a.draft_format,wide_m04_requested=a.wide_m04,preferred_serving_context=192000,conditionally_acceptable_serving_context=166000,context_fallback_requires_clear_speed_advantage=True,low_context_experiment_authorized=True,baseline_regenerated=False,attempts=[],started_unix=time.time(),sources={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
def api(path,data=None,timeout=180):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
def metrics():
    with urllib.request.urlopen(BASE+'/metrics',timeout=30) as response:raw=response.read().decode()
    result={}
    for line in raw.splitlines():
        match=re.match(r'([^#{ ]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)$',line)
        if match:result[match[1]]=result.get(match[1],0)+float(match[2])
    return result
class DraftWorker(Worker):
    stopped=False
    def stop(self):
        if not self.stopped:
            super().stop();self.stopped=True
    def __init__(self,context,folder):
        snapshot=folder/'source-snapshot';snapshot.mkdir(parents=True,exist_ok=True)
        for source in [Path(__file__).parent]+([Path(__file__).parent.parent/'dflash2-pro-diagnostics-v031'] if a.capture_q or a.draft_format=='w4' else [])+([Path(__file__).parent.parent/'dflash2-w4-v031'] if a.draft_format=='w4' else []):
            for file in source.glob('*.py'):
                target=snapshot/file.name;assert not target.exists(),file.name;target.write_bytes(file.read_bytes())
        binds=[(snapshot,'/experiment')]+([(draft,'/draft')] if a.method=='dflash' else [])
        super().__init__(image,folder,'b70-dflash2-serving-test',env={'EXL3_SDPA_CACHE_CAPACITY':'64','VLLM_SERVER_DEV_MODE':'1','PYTHONPATH':'/experiment:/opt/b70-runtime'},binds=binds)
        args=self.profile['argv'];args[args.index('--max-model-len')+1]=str(context)
        spec_config=dict(method=a.method,num_speculative_tokens=a.depth)
        if a.method=='dflash':spec_config.update(model='/draft',draft_sample_method=a.draft_sample_method)
        args[args.index('--speculative-config')+1]=json.dumps(spec_config)
        extension='mtp_inspect.WorkerExtension' if a.method=='mtp' else ('wide_inspect.WorkerExtension' if a.wide_m04 else 'dflash_inspect.WorkerExtension')
        if a.draft_format=='w4':extension='w4_inspect.WorkerExtension'
        if a.capture_q:
            assert a.method=='dflash' and a.wide_m04
            extension='w4_inspect.WorkerExtension' if a.draft_format=='w4' else 'diagnostics.WorkerExtension'
        args+=['--worker-extension-cls',extension]
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);w=None;save()
    try:
        subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
        assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()==['b70-chatui']
        for context in map(int,a.contexts.split(',')):
            w=DraftWorker(context,root/f'context-{context}');attempt=dict(max_model_len=context,status='STARTING');state['attempts'].append(attempt);save();print('START_CONTEXT',context,flush=True)
            try:w.start()
            except BaseException as error:
                w.stop();log=(w.root/'worker.log').read_text();attempt.update(status='FAILED_STARTUP',error=repr(error));save()
                if any(x in log for x in ['maximum sequence length','estimated maximum model length','try increasing `gpu_memory_utilization`','No available memory for the cache blocks','max seq len']):
                    attempt['status']='INSUFFICIENT_CONTEXT_CAPACITY';save();print('CAPACITY_REJECTED',context,flush=True);w=None;continue
                raise
            attempt['status']='READY';save();print('READY_CONTEXT',context,flush=True)
            identity=api('/collective_rpc',dict(method='inspect_mtp_setup' if a.method=='mtp' else 'inspect_dflash_setup',args=[],kwargs={}),timeout=240)
            (root/'worker-attestation.json').write_text(json.dumps(identity,indent=2)+'\n')
            audits=list((w.root/'loader').glob('loader-*.json'));assert len(audits)==1
            mtp_modules=8 if a.method=='mtp' else 0
            audit=json.loads(audits[0].read_text());assert audit['status']=='PASS' and audit['expected_modules']==audit['loaded_modules']==401+mtp_modules and audit['mtp_expected']==audit['mtp_loaded']==mtp_modules
            if a.draft_format=='w4':
                ledger=api('/collective_rpc',dict(method='cache_ledger',args=[],kwargs={}),timeout=900)
                (root/'cache-ledger-rpc.json').write_text(json.dumps(ledger,indent=2)+'\n')
                numerical=api('/collective_rpc',dict(method='w4_numerical_probe',args=[],kwargs={}),timeout=900)
                (root/'w4-numerical.json').write_text(json.dumps(numerical,indent=2)+'\n')
                print('PASS_W4_NUMERICAL_PROBE',flush=True)
            attempt['worker_attested']=True;save()
            api('/reset_prefix_cache',{})
            before=metrics()
            body=dict(model='Qwen3.8-27B',messages=[dict(role='user',content='Write a Python function that adds two integers, with one example.')],temperature=0,max_tokens=256,chat_template_kwargs={'enable_thinking':False})
            cold=api('/v1/chat/completions',body,timeout=240);warm=api('/v1/chat/completions',body,timeout=240);after=metrics()
            delta={k:after.get(k,0)-v for k,v in before.items()}
            accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0);drafted=delta.get('vllm:spec_decode_num_draft_tokens_total',0)
            result=dict(cold=cold,warm=warm,accepted=accepted,drafted=drafted)
            (root/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
            assert accepted>0 and drafted>0,(accepted,drafted)
            assert all('def ' in item['choices'][0]['message']['content'] for item in [cold,warm])
            print('PASS_NONZERO_ACCEPTANCE',accepted,drafted,flush=True)
            prefix_cmd=[sys.executable,str(REPO/'scripts/check-prefix-state.py'),'--root',BASE,'--output',str(root/'prefix-state.json')]
            with (root/'prefix-state.log').open('w') as log:subprocess.run(prefix_cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            print('PASS_PREFIX_STATE',flush=True)
            if a.greedy_control:
                controls=[]
                for label in ['phase-32k-c1-r1','phase-64k-c1-r1']:
                    body=dict(model='Qwen3.8-27B',messages=[dict(role='user',content=(fixture/label/'prompt-1.txt').read_text())],
                        temperature=0,max_tokens=256,ignore_eos=True,return_token_ids=True,
                        chat_template_kwargs={'enable_thinking':False})
                    api('/reset_prefix_cache',{})
                    responses=[]
                    for mode in ['cold','warm']:
                        time.sleep(1)  # untimed periodic-metrics flush
                        before=metrics();started=time.monotonic()
                        response=api('/v1/chat/completions',body,timeout=900)
                        wall=time.monotonic()-started
                        assert response['usage']['completion_tokens']==256,response['usage']
                        ids=response['choices'][0]['token_ids'];assert len(ids)==256
                        # Wait for periodic metric accounting after the last
                        # token before recording this isolated wave.
                        deadline=time.monotonic()+45
                        while True:
                            after=metrics()
                            generated=after.get('vllm:generation_tokens_total',0)-before.get('vllm:generation_tokens_total',0)
                            completed=after.get('vllm:request_success_total',0)-before.get('vllm:request_success_total',0)
                            decode=after.get('vllm:request_decode_time_seconds_sum',0)-before.get('vllm:request_decode_time_seconds_sum',0)
                            if generated==256 and completed==1 and decode>0:break
                            assert time.monotonic()<deadline,generated
                            time.sleep(.25)
                        delta={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after)}
                        assert delta.get('vllm:num_preemptions_total',0)==0
                        responses.append(dict(mode=mode,usage=response['usage'],token_ids=ids,
                            wall_s=wall,native_decode_tps=255/decode,metrics_delta=delta))
                    assert responses[0]['token_ids']==responses[1]['token_ids'],'Greedy cold/warm state disagreement'
                    controls.append(dict(fixture=label,responses=responses))
                (root/'greedy-control.json').write_text(json.dumps(dict(status='PASS_GREEDY_FROZEN_COLD_WARM',controls=controls),indent=2)+'\n')
                print('PASS_GREEDY_FROZEN_CONTROL',flush=True)
            if a.capture_q:
                for method in ['install_round_capture','begin_round_capture']:
                    api('/collective_rpc',dict(method=method,args=[],kwargs={}))
                capture_body=dict(model='Qwen3.8-27B',messages=[dict(role='user',content=(fixture/'phase-32k-c1-r1/prompt-1.txt').read_text())],temperature=1.0,top_p=.95,top_k=20,seed=770000,max_tokens=384,ignore_eos=True,chat_template_kwargs={'enable_thinking':False})
                api('/reset_prefix_cache',{})
                capture=api('/v1/chat/completions',capture_body,timeout=900)
                assert capture['usage']['completion_tokens']==384
                (root/'capture-response.json').write_text(json.dumps(capture,indent=2)+'\n')
                for method in ['finish_round_capture','uninstall_round_capture']:
                    receipt=api('/collective_rpc',dict(method=method,args=[],kwargs={}))
                    (root/(method+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                print('PASS_CAPTURE_AND_REMOVAL_OF_Q_HOOKS',flush=True)
            if a.benchmark:
                names=a.scenario_names.split(',')
                if context<65536+1024:names=[name for name in names if name!='phase-64k-c1']
                scenarios=[s for s in json.loads((REPO/'benchmarks/current-profile-scenarios.json').read_text()) if s['name'] in names]
                assert len(scenarios)==len(names) and scenarios
                for s in scenarios:s['repeats']=3
                path=root/'scenarios.json';path.write_text(json.dumps(scenarios,indent=2)+'\n')
                command=[sys.executable,str(REPO/'scripts/current-profile-benchmark.py'),'--base',BASE,'--container',w.name,'--expected-max-num-seqs','16','--fixture-root',str(fixture),'--legacy-prefix-namespace','--scenarios',str(path),'--output-root',str(root/'speed'),'--execute']
                state['benchmark_command']=command;save();print('START_FROZEN_SPEED_SCREEN',names,flush=True)
                with (root/'benchmark.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                paths=list((root/'speed').glob('*/results.json'));assert len(paths)==1
                raw=json.loads(paths[0].read_text());assert len(raw['cases'])==len(scenarios)*3
                assert all(not c['preemptions'] and all(q['usage']['completion_tokens']==1024 for q in c['requests']) for c in raw['cases'])
                state['benchmark_results']=str(paths[0].relative_to(REPO));state['benchmark_sha256']=sha(paths[0]);print('PASS_FROZEN_SPEED_SCREEN',flush=True)
            if a.capacity_probe:
                assert context>=64000
                api('/reset_prefix_cache',{})
                prompt=(fixture/'phase-4k-c1-r1/prompt-1.txt').read_text()
                tokenized=api('/tokenize',dict(model='Qwen3.8-27B',messages=[dict(role='user',content=prompt)],chat_template_kwargs={'enable_thinking':False}))
                ids=tokenized['tokens'];assert len(ids)==tokenized['count']
                prompt_len=context-1024
                long_ids=(ids*((prompt_len+len(ids)-1)//len(ids)))[:prompt_len]
                before=metrics();started=time.monotonic();print('START_EXACT_CONTEXT_PROBE',context,flush=True)
                result=api('/v1/completions',dict(model='Qwen3.8-27B',prompt=long_ids,max_tokens=1024,ignore_eos=True,temperature=0),timeout=3600)
                after=metrics();usage=result['usage']
                assert usage['prompt_tokens']==prompt_len and usage['completion_tokens']==1024 and usage['total_tokens']==context,usage
                assert result['choices'][0]['text'].strip() and result['choices'][0]['finish_reason']=='length'
                assert after.get('vllm:num_preemptions_total',0)==before.get('vllm:num_preemptions_total',0)
                deadline=time.monotonic()+45
                while True:
                    computed=after['vllm:request_prefill_kv_computed_tokens_sum']-before.get('vllm:request_prefill_kv_computed_tokens_sum',0)
                    if computed>=prompt_len:break
                    if time.monotonic()>deadline:raise RuntimeError('Capacity metrics did not settle')
                    time.sleep(.25);after=metrics()
                assert computed==prompt_len,('Unexpected prefill/recompute accounting',computed,prompt_len)
                receipt=dict(status='PASS_EXACT_CONTEXT_CAPACITY_NOT_SEMANTIC_QUALITY',usage=usage,prefill_computed_tokens=computed,unexpected_recompute_tokens=0,wall_s=time.monotonic()-started,accepted=after.get('vllm:spec_decode_num_accepted_tokens_total',0)-before.get('vllm:spec_decode_num_accepted_tokens_total',0),drafted=after.get('vllm:spec_decode_num_draft_tokens_total',0)-before.get('vllm:spec_decode_num_draft_tokens_total',0),prompt_token_ids_sha256=hashlib.sha256(json.dumps(long_ids).encode()).hexdigest(),source='cycled frozen source-review token IDs; capacity-only fixture',response=result)
                (root/'exact-context.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print('PASS_EXACT_CONTEXT_PROBE',context,flush=True)
                # End-state sampling and custom-stop handling share a frozen
                # near-limit prompt; this is separate from speed measurement.
                end_prompt=long_ids+ids[:1008]
                assert len(end_prompt)==context-16
                api('/reset_prefix_cache',{})
                end_body=dict(model='Qwen3.8-27B',prompt=end_prompt,max_tokens=16,ignore_eos=True,
                    temperature=1.,top_p=.95,top_k=20,seed=770000,return_token_ids=True)
                before=metrics();end_sample=api('/v1/completions',end_body,timeout=3600)
                assert end_sample['usage']['total_tokens']==context and end_sample['usage']['completion_tokens']==16
                tokens=end_sample['choices'][0]['token_ids'];assert len(tokens)==16
                stop_sample=api('/v1/completions',{**end_body,'ignore_eos':False,'stop_token_ids':[tokens[0]]},timeout=3600)
                assert stop_sample['choices'][0]['finish_reason']=='stop' and stop_sample['usage']['completion_tokens']==1,stop_sample
                after=metrics();assert after.get('vllm:num_preemptions_total',0)==before.get('vllm:num_preemptions_total',0)
                (root/'context-end-state.json').write_text(json.dumps(dict(status='PASS_STOCHASTIC_CONTEXT_END_AND_CUSTOM_STOP',
                    sampling=end_sample,custom_stop=stop_sample,preemptions=0,scope='Near-limit sampling and custom stop; not a general semantic quality score'),indent=2)+'\n')
                print('PASS_CONTEXT_END_STATE',context,flush=True)
            state['active_context_limit']=context
            state['status']=('PASS_CHECKPOINT_NATIVE_MTP_STARTUP_ACCEPTANCE_AND_SPEED' if a.method=='mtp' else ('PASS_W4_DFLASH_DIAGNOSTIC_SCREEN' if a.draft_format=='w4' else 'PASS_UNQUANTIZED_DFLASH_STARTUP_ACCEPTANCE_AND_SPEED')) if a.benchmark else 'PASS_STARTUP_AND_ACCEPTANCE'
            save();break
        else:state['status']='NO_CONTEXT_CANDIDATE_FITS';raise RuntimeError('No authorized test context fits')
    except BaseException as error:state.update(status='FAILED',error=repr(error));raise
    finally:
        if w is not None:w.stop()
        subprocess.run(['systemctl','--user','start','b70-qwen38-vllm.service'],check=True,timeout=90)
        state['restoration']=restore.wait_for_qualified_service(production,timeout=900)
        state['finished_unix']=time.time();save()
