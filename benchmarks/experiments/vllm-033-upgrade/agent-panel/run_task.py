#!/usr/bin/env python3
"""One frozen genuine coding task with production Thinking budget and tools."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

spec=importlib.util.spec_from_file_location('coding_tools',Path(__file__).with_name('coding_tools.py'))
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
K.METRICS['verify_rounds']='vllm:spec_decode_num_drafts_total'


def isolated(project,image,suite):
    command=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
        '--user',str(os.getuid())+':'+str(os.getgid()),
        '--pids-limit','64','--memory','512m','--cpus','2','--tmpfs','/tmp:rw,nosuid,nodev',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/workspace',
        '-v',str(project.resolve())+':/workspace:ro','-w','/workspace']
    if suite is not None:command+=['-v',str(suite.parent.resolve())+':/acceptance:ro']
    command+=['--entrypoint','python',image,'-m','unittest','discover','-s','/acceptance' if suite else 'tests','-v']
    if suite:command+=['-p',suite.name]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45);raw=result.stdout+result.stderr
    count=re.search(r'Ran (\d+) tests? in ',raw)
    if count is None or int(count.group(1))==0:raise RuntimeError('Invalid independent test harness: '+raw)
    return dict(passed=result.returncode==0,exit_code=result.returncode,tests_run=int(count.group(1)),output=raw)


def main():
    p=argparse.ArgumentParser()
    for name in ['fixture','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--base',required=True);p.add_argument('--container',required=True);p.add_argument('--image',required=True)
    p.add_argument('--identity-receipt',type=Path,required=True);p.add_argument('--launch-receipt',type=Path,required=True)
    p.add_argument('--runtime-source',type=Path,required=True);p.add_argument('--worker-source',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--wall-seconds',type=int,default=900);a=p.parse_args()
    manifest=json.loads((a.fixture/'manifest.json').read_text());assert a.seed in manifest['seeds']
    actual={str(f.relative_to(a.fixture)):K.sha(f) for f in a.fixture.rglob('*') if f.is_file() and f.name!='manifest.json'}
    assert actual==manifest['files'],'Frozen fixture changed'
    a.out.mkdir(exist_ok=False,parents=True);project=a.out/'project';shutil.copytree(a.fixture/'project',project)
    captures=a.out/'captures';captures.mkdir();identity=K.live_identity(a.container)
    receipt=json.loads(a.identity_receipt.read_text())
    assert identity['image_id']==receipt['image_id']==a.image
    assert identity['container_id']==receipt['container_id']
    launch=json.loads(a.launch_receipt.read_text())
    assert launch['profile']['config']['vllm']['max_num_seqs']==16
    runtime_hashes={f.name:K.sha(f) for f in a.runtime_source.glob('*.py')}
    fix_sha=receipt['cow_fix_sha256']
    source_identity=dict(candidate_cow_fix_sha256=fix_sha,coding_tools_sha256=K.sha(Path(K.__file__)),image_id=identity['image_id'],native_binary_sha256=receipt['native_binary_sha256'],
        runtime_source_sha256=runtime_hashes,worker_launcher_sha256=K.sha(a.worker_source),
        extension_source_sha256=runtime_hashes)
    tools=json.loads(json.dumps(K.TOOLS));tools[0]['function']['description']='List every file in the current coding project.'
    messages=[{'role':'system','content':'You are a software engineer working in a small Python repository. Use tools to inspect files, implement the requested change and run visible tests. Make the smallest complete implementation. When done, return a concise final message without tool calls. Hidden acceptance tests are not available.'},
        {'role':'user','content':(a.fixture/'instruction.txt').read_text()}]
    records=[];tool_count=0;repair_count=0;failed_visible=False;before=K.snapshot(a.base);started=time.monotonic();complete=False
    summary=dict(status='RUNNING',fixture_id=manifest['fixture_id'],fixture_manifest_sha256=K.sha(a.fixture/'manifest.json'),
        fixture_project_sha256=K.tree_hash(a.fixture/'project'),identity=identity,seed=a.seed,
        image_id=identity['image_id'],native_binary_sha256=receipt['native_binary_sha256'],
        source_contract_sha256=hashlib.sha256(json.dumps(source_identity,sort_keys=True).encode()).hexdigest(),
        source_identity=source_identity,profile_hash=hashlib.sha256(json.dumps(launch['profile'],sort_keys=True).encode()).hexdigest(),
        slots=launch['profile']['config']['vllm']['max_num_seqs'],power_w=180,ram_cache_gib=12,
        runner_sha256=K.sha(Path(__file__)),thinking_token_budget=8192,source_contract='Frozen identical task/profile, real tools, official regular Thinking sampling; natural model endings, no hidden feedback or manual repairs.')
    def save():(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save()
    try:
        for step in range(24):
            remaining=a.wall_seconds-(time.monotonic()-started)
            if remaining<30:raise TimeoutError('Frozen agent task wall limit')
            payload=dict(model=K.MODEL,messages=messages,tools=tools,tool_choice='auto',
                temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,repetition_penalty=1.,frequency_penalty=0.,
                seed=a.seed+step,max_tokens=16384,thinking_token_budget=8192,cache_salt=manifest['fixture_id']+str(a.seed),
                chat_template_kwargs={'enable_thinking':True,'reasoning_effort':'medium','preserve_thinking':True})
            rendered_messages=[]
            for original in messages:
                message=dict(original);reasoning=message.pop('reasoning_content',None)
                if reasoning is not None and message.get('reasoning') is None:message['reasoning']=reasoning
                rendered_messages.append(message)
            rendered=K.http_json(a.base+'/tokenize',dict(model=K.MODEL,messages=rendered_messages,tools=tools,
                chat_template_kwargs=payload['chat_template_kwargs'],add_generation_prompt=True),120)
            available=262144-rendered['count'];assert available>=512,'Common short-task history window exhausted'
            payload['max_tokens']=min(16384,available)
            (captures/f'request-{step:03d}.json').write_text(json.dumps(payload,indent=2)+'\n')
            start=time.monotonic();request_before=K.snapshot(a.base)
            response=K.http_json(a.base+'/v1/chat/completions',payload,min(900,max(30,int(remaining))))
            request_wall=time.monotonic()-start;request_after=K.snapshot(a.base)
            (captures/f'response-{step:03d}.json').write_text(json.dumps(response,indent=2)+'\n')
            assert response['usage']['prompt_tokens']==rendered['count']
            choice=response['choices'][0];message=choice['message'];calls=message.get('tool_calls') or []
            records.append(dict(step=step,wall_s=request_wall,rendered_tokens=rendered['count'],usage=response['usage'],
                finish_reason=choice['finish_reason'],tool_names=[c['function']['name'] for c in calls],
                native_delta={k:request_after[k]-request_before[k] for k in request_before},metrics=response.get('metrics')))
            messages.append(message)
            for call in calls:
                tool_count+=1;name=call['function']['name']
                try:
                    arguments=json.loads(call['function'].get('arguments') or '{}')
                    if name=='run_tests':
                        result=isolated(project,a.image,None);failed_visible|=not result['passed']
                        result['output']=re.sub(r'Ran (\d+) tests? in [0-9.]+s',r'Ran \1 tests in <elapsed>s',result['output'])
                    else:
                        result=K.execute_tool(project,name,arguments)
                        if name=='write_file' and failed_visible:repair_count+=1
                except Exception as exc:result={'error':type(exc).__name__+': '+str(exc)}
                (captures/f'tool-{tool_count:03d}.json').write_text(json.dumps(dict(name=name,result=result),indent=2)+'\n')
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
            (a.out/'requests.json').write_text(json.dumps(records,indent=2)+'\n')
            print('AGENT_STEP',step,choice['finish_reason'],[c['function']['name'] for c in calls],flush=True)
            if not calls and choice['finish_reason']!='length':complete=True;break
            if not calls:messages.append({'role':'user','content':'Continue the implementation and use tools as needed.'})
        wall=time.monotonic()-started;after=K.snapshot(a.base);delta={k:after[k]-before[k] for k in before}
        assert {f.name:K.sha(f) for f in a.runtime_source.glob('*.py')}==runtime_hashes,'Host runtime changed during task'
        # Independent hidden grading occurs after the taskwall timer, never fed
        # to the agent and never used to grant a repair continuation.
        check=isolated(project,a.image,a.fixture/'acceptance/test_acceptance.py')
        full=complete and check['passed']
        output_tokens=sum(r['usage']['completion_tokens'] for r in records)
        if output_tokens>len(records):assert delta['verify_rounds']>0,'Native verify counter missing'
        summary.update(status='COMPLETED_AGENT_TASK',full_success=full,acceptance=check,task_wall_s=wall,
            time_to_success_s=wall if full else None,requests=len(records),tool_calls=tool_count,
            repair_write_count_after_failed_visible_test=repair_count,generated_tokens=output_tokens,
            native_delta=delta,native_decode_tps=(output_tokens-len(records))/delta['decode_seconds'] if delta['decode_seconds'] else None,
            max_input_tokens=max((r['rendered_tokens'] for r in records),default=0),
            context_max=max((r['usage']['total_tokens'] for r in records),default=0),
            actual_verify_rounds=delta['verify_rounds'],emitted_tokens=output_tokens,
            raw_verify_emissions=delta['verify_rounds']+delta['accepted_tokens'],
            verify_round_note='Actual native spec_decode_num_drafts counter; raw terminal verify emissions can exceed visible API tokens.',
            project_sha256=K.tree_hash(project),natural_end=complete)
    except BaseException as exc:summary.update(status='FAILED_HARNESS_OR_LIMIT',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
