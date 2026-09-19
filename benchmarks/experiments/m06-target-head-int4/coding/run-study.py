#!/usr/bin/env python3
"""Pinned real-agent A/B, with per-request counters and independent evaluation."""
import argparse,fcntl,hashlib,importlib.util,json,os,re,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/sebastian/LocalLLM/intel-b70-qwen38-vllm')
spec=importlib.util.spec_from_file_location('diag',REPO/'scripts/run-diagnostics.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
NAME='b70-m06-target-head-arm'

def ratio(a,b):return a/b if b else None

def aggregate(rows):
    good=[r for r in rows if r['status']=='ok']
    total=lambda key:sum(r[key] for r in good)
    return {'requests':len(rows),'valid_requests':len(good),'actual_context_min':min((r['prompt_tokens'] for r in good),default=None),'actual_context_max':max((r['prompt_tokens'] for r in good),default=None),'prompt_tokens':total('prompt_tokens'),'generated_tokens':total('generated_tokens'),'computed_tokens':total('computed_tokens'),'cached_tokens':total('prefix_hit_tokens'),'prefill_seconds':total('native_prefill_s'),'decode_seconds':total('native_decode_s'),'prefill_compute_tps':ratio(total('computed_tokens'),total('native_prefill_s')),'decode_tps':ratio(total('generated_tokens')-len(good),total('native_decode_s')),'prefix_hit_rate':ratio(total('prefix_hit_tokens'),total('prefix_query_tokens')),'mtp_acceptance':ratio(total('accepted_tokens'),total('drafted_tokens'))}

def summarize(run,state):
    items=[]
    for arm in state['arms']:
        out=Path(arm['directory']);p=out/'task-result.json'
        if not p.exists():continue
        task=json.loads(p.read_text());rows=task['requests'];value={'arm':arm['label'],'image':arm['image'],'task_status':task['status'],'failure_category':task['failure_category'],'wall_seconds':task['wall_s'],'tool_seconds':task['tool_time_s'],'compactions':task['compactions'],'overall':aggregate(rows),'bands':[],'validation':arm.get('validation')}
        for band in sorted({int(r['prompt_tokens']//10000)*10000 for r in rows if r['status']=='ok'}):value['bands'].append({'band':f'{band}-{band+10000}',**aggregate([r for r in rows if r.get('prompt_tokens',-1)//10000==band//10000])})
        telemetry=out/'telemetry.jsonl'
        if telemetry.exists():
            ts=[json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()];energy=[x for x in ts if 'energy_uj' in x]
            if len(energy)>1:
                joules=(energy[-1]['energy_uj']-energy[0]['energy_uj'])/1e6;seconds=(energy[-1]['monotonic_ns']-energy[0]['monotonic_ns'])/1e9
                value['card_energy']={'scope':'complete agent session, includes tool/idle time','joules':joules,'sampled_seconds':seconds,'average_watts':ratio(joules,seconds),'generated_tokens_per_joule':ratio(value['overall']['generated_tokens'],joules),'power_caps_w':sorted({x['power_cap_uw']/1e6 for x in energy})}
        items.append(value)
    summary={'status':state['status'],'method':'paired adaptive coding trajectories, not identical-request replay','arms':items};diag.save(run/'summary.json',summary)
    lines=['# M06 real-agent coding A/B','',f"Status: {state['status']}",'','Both profiles: 180 W, same source and task; adaptive trajectories may differ.','', '| Profile | Task status | Context | Requests | Prefill tok/s | Decode tok/s |','|---|---|---|---:|---:|---:|']
    def fmt(v):return f'{v:.2f}' if isinstance(v,(float,int)) else 'unmeasured'
    for a in items:
        for b in a['bands']:lines.append(f"| {a['arm']} | {a['task_status']} | {b['band']} | {b['valid_requests']} | {fmt(b['prefill_compute_tps'])} | {fmt(b['decode_tps'])} |")
    lines+=['','See summary.json for task outcomes, phase totals, cache hits, MTP acceptance, energy and independent validation. Unvisited bands are unmeasured. Raw agent output remains local.','']
    (run/'RESULTS.md').write_text('\n'.join(lines))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();run=args.run_dir.resolve();manifest=json.loads((run/'manifest.json').read_text())
    state={'status':'preflight','started_at':time.time(),'arms':[]};save=lambda:diag.save(run/'state.json',state);save()
    lock=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    production=diag.production_inspect()
    if production['Image']!=manifest['control'] or int(diag.POWER_CAP.read_text())!=180000000:raise RuntimeError('production image/power changed')
    diag.assert_mtp4(production['Config']['Cmd']);diag.ensure_idle(diag.PRODUCTION_URL)
    if not json.loads((run/'preflight.json').read_text())['ready']:raise RuntimeError('preflight not ready')
    for label in ['control','candidate']:
        actual=subprocess.check_output(['docker','image','inspect',manifest[label],'--format','{{.Id}}'],text=True).strip()
        if actual!=manifest[label]:raise RuntimeError('image mismatch')
    diag.save(run/'production-inspect.json',production);(run/'production-owned').write_text('exclusive benchmark owner\n')
    engine=None;agent=None
    try:
        subprocess.run(['systemctl','--user','stop',diag.PRODUCTION_SERVICE],check=True,timeout=120)
        for index,label in enumerate(['control','candidate']):
            out=run/f'arm-{index}-{label}';out.mkdir();shutil.copytree(run/'frozen',out/'workspace',symlinks=True)
            # Each workspace is its own repository; hooks and live git metadata are absent.
            for cmd in [['git','init','-q'],['git','add','dashboard'],['git','-c','user.name=Benchmark','-c','user.email=benchmark@localhost','commit','-qm','Frozen coding baseline']]:subprocess.run(cmd,cwd=out/'workspace',check=True,stdout=subprocess.DEVNULL)
            arm={'label':label,'image':manifest[label],'directory':str(out)};state['arms'].append(arm);state['status']=f'loading-{label}';save()
            command=diag.engine_command(production,name=NAME,image=manifest[label],evidence=out,arguments=production['Config']['Cmd'].copy());command[command.index(manifest[label]):command.index(manifest[label])]=['-e','VLLM_SERVER_DEV_MODE=1'];diag.save(out/'engine-command.json',command)
            with (out/'engine.log').open('w') as log:
                engine=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);diag.wait_ready(engine)
                if diag.api('/v1/models')['data'][0]['max_model_len']!=200704:raise RuntimeError('context changed')
                if label=='candidate':
                    text=(out/'engine.log').read_text()
                    if text.count('B70_TARGET_LMHEAD_INT4_READY')!=1 or 'source_numel 0' not in text:raise RuntimeError('target head not finalized exactly once')
                state['status']=f'warmup-{label}';save()
                with (out/'warmup.log').open('w') as warm:
                    subprocess.run(['python3',str(REPO/'scripts/benchmark.py'),'--root',diag.EXPERIMENT_URL,'--contexts','8192','--output-tokens','64','--repeats','1','--nonce','excluded-coding-warmup','--output-dir',str(out/'warmup')],check=True,timeout=900,stdout=warm,stderr=subprocess.STDOUT)
                reset=diag.api('/reset_prefix_cache',{});diag.save(out/'cache-reset.json',reset)
                if reset!={'success':True}:raise RuntimeError('prefix cache reset failed')
                diag.ensure_idle(diag.EXPERIMENT_URL)
                diag.save(out/'agent-boundary-before.json',diag.prometheus()[0]);state['status']=f'coding-{label}';save()
                with diag.TelemetrySampler(out/'telemetry.jsonl') as sampler, (out/'driver.log').open('w') as driver:
                    agent=subprocess.Popen(['node',str(ROOT/'run-task.mjs'),str(out),str(run),label],stdout=driver,stderr=subprocess.STDOUT)
                    try:arm['agent_exit_code']=agent.wait(timeout=manifest['limits']['wall_seconds']+90)
                    except subprocess.TimeoutExpired:
                        agent.terminate();agent.wait(timeout=30);arm['agent_exit_code']='supervisor-timeout'
                    finally:agent=None
                arm['telemetry_errors']=sampler.errors
                diag.save(out/'agent-boundary-after.json',diag.prometheus()[0]);save();summarize(run,state)
                diag.stop_container(NAME,engine);engine=None
    except BaseException as e:
        state['status']='infrastructure-failed';state['error']=repr(e);save();raise
    finally:
        if agent is not None:agent.terminate()
        diag.stop_container(NAME,engine)
        subprocess.run(['python3',str(ROOT/'recover.py'),'--run-dir',str(run)],check=True,timeout=900)
        save();summarize(run,state)
    # Evaluate only after freeing the GPU and restoring production.
    state['status']='independent-evaluation';save()
    for arm in state['arms']:
        out=Path(arm['directory']);checks=json.loads((ROOT/'baseline-checks.json').read_text())
        with (out/'validation.log').open('w') as log:
            p=subprocess.run(['node',str(ROOT/'check-workspace.mjs'),str(out/'workspace'),str(run/'dependencies'),json.dumps(checks),'continue'],stdout=log,stderr=subprocess.STDOUT,timeout=1250)
        with (out/'acceptance.log').open('w') as log:
            a=subprocess.run(['node',str(ROOT/'check-workspace.mjs'),str(out/'workspace'),str(run/'dependencies'),json.dumps([['node','/runtime/acceptance.mjs']])],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        validation={'suite_exit_code':p.returncode,'acceptance_exit_code':a.returncode,'baseline_format_failures':manifest['baseline_format_failures']}
        text=(out/'acceptance.log').read_text();pos=text.find('{\n  "schema": 1,')
        if pos>=0:validation['acceptance']=json.JSONDecoder().raw_decode(text[pos:])[0]
        clean=re.sub(r'\x1b\[[0-9;]*m','',(out/'validation.log').read_text());warnings=[line.split('[warn] ',1)[1] for line in clean.splitlines() if '[warn] ' in line and 'Code style issues' not in line]
        validation['format_warnings']=warnings
        validation['only_preexisting_format_failure']=p.returncode==1 and clean.count('CHECK_FAILED')==1 and set(warnings)<=set(manifest['baseline_format_failures']) and len(warnings)>0
        validation['checks_pass']=p.returncode==0 or validation['only_preexisting_format_failure'];validation['task_acceptance_pass']=a.returncode==0
        arm['validation']=validation
        with (out/'result.diff').open('w') as f:subprocess.run(['git','-c','core.hooksPath=/dev/null','diff','HEAD','--','dashboard'],cwd=out/'workspace',stdout=f,check=True)
        arm['untracked_files']=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=out/'workspace',text=True).splitlines()
        save();summarize(run,state)
    state['status']='complete-not-promoted';state['finished_at']=time.time();save();summarize(run,state)

if __name__=='__main__':main()
