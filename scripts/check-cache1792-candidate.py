"""Bounded release-child check; original production restored on every exit."""
import argparse,datetime as dt,fcntl,importlib.util,json,os,re,signal,subprocess,time,urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
from exl3_candidate_worker import profile,REPO
from ram_kv_release import load_overlay,serving_arguments

def main():
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
    now=dt.datetime.now(ZoneInfo('Europe/Berlin'));assert now<now.replace(hour=22,minute=20,second=0,microsecond=0)
    root=a.out.resolve();root.mkdir(exist_ok=False)
    release=json.loads((a.release_dir/'production_image.json').read_text());image=release['image_id'];cfg=profile(image);overlay=load_overlay(REPO,release,{})
    argv=serving_arguments(cfg['argv'][1:],overlay)+['--enable-per-request-metrics','--enable-prompt-tokens-details']
    argv[argv.index('--host')+1]='127.0.0.1';argv[argv.index('--port')+1]='8039'
    assert '--worker-extension-cls' not in argv and '--prefix-match-unit' in argv
    name='b70-cache1792-release-gate';state=dict(status='STARTING',image_id=image,started_Berlin=now.isoformat())
    def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
    def interrupted(sig,frame):raise InterruptedError(str(sig))
    for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
    spec=importlib.util.spec_from_file_location('cache1792_api',REPO/'scripts/check-cache1792-api.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    with Path('/tmp/b70-v033-qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stopped=False;started=False;signal.alarm(1200);save()
        try:
            assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
            stopped=True;subprocess.run(['systemctl','--user','stop','b70-qwen38-vllm.service'],check=True,timeout=90)
            cache=Path.home()/'.cache/b70-qwen38-vllm-production'/release['policy_sha256']/image.removeprefix('sha256:')
            cmd=['docker','run','-d','--name',name,'--network=host','--device=/dev/dri','--shm-size=14g','--memory=23g','--memory-swap=23g','-v',str(Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw')+':/models/checkpoint:ro','-v',str(REPO/'runtime')+':/opt/b70-runtime:ro','-v',str(root)+':/results']
            for key,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
                path=cache/key;path.mkdir(parents=True,exist_ok=True);cmd+=['-v',str(path)+':'+target]
            env={**cfg['env'],**overlay['environment'],'HF_HUB_OFFLINE':'1','PYTHONPATH':'/opt/b70-runtime','ZE_AFFINITY_MASK':'0','EXL3_LOADER_REPORT_DIR':'/results/loader'}
            for key,value in env.items():cmd+=['-e',key+'='+value]
            cmd+=['--entrypoint','/opt/venv/bin/vllm',image,*argv]
            (root/'launch.json').write_text(json.dumps(dict(command=cmd,image_id=image),indent=2)+'\n')
            subprocess.run(cmd,check=True,capture_output=True,timeout=60);started=True
            deadline=time.monotonic()+720
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8039/health',timeout=3) as f:assert f.status==200
                    break
                except OSError:
                    assert time.monotonic()<deadline
                    info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];assert info['State']['Running'];time.sleep(2)
            logs=subprocess.run(['docker','logs',name],capture_output=True,text=True,check=True);text=logs.stdout+logs.stderr
            match=re.search(r'B70_COMPLETE_COW_READY (\{[^\n]+\})',text);assert match,'Permanent COW activation missing'
            receipt=json.loads(match.group(1));assert receipt['original_views']==65 and receipt['selected_views']==17
            assert receipt['patch_sha256']=='b30646e39778904cb3f5ce2664d37defe7a743b8a4e6715d5b883f1ec54d5f43'
            (root/'automatic-cow-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            result=api.run('http://127.0.0.1:8039',root/'api',a.fixture)
            kernel=subprocess.check_output(['journalctl','-k','--since',now.strftime('%Y-%m-%d %H:%M:%S'),'--no-pager'],text=True);(root/'kernel.log').write_text(kernel)
            assert not re.search('Engine memory CAT error|Fault response:.*EBUSY|Timedout job',kernel)
            state.update(status='PASS_EXACT_RELEASE_CHILD_AUTOMATIC_COW_AND_REAL_API',api_status=result['status'])
        except BaseException as exc:state.update(status='FAILED_OR_INTERRUPTED',error=repr(exc));raise
        finally:
            signal.alarm(0)
            if started:
                logs=subprocess.run(['docker','logs',name],capture_output=True,text=True);(root/'worker.log').write_text(logs.stdout+logs.stderr)
                subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=60)
            if stopped:
                result=subprocess.run(['python3',str(REPO/'scripts/restore-production.py')],capture_output=True,text=True,timeout=900);(root/'restore.log').write_text(result.stdout+result.stderr);state['restore_returncode']=result.returncode
            state['finished_Berlin']=dt.datetime.now(ZoneInfo('Europe/Berlin')).isoformat();save()
    print(json.dumps(state))

if __name__=='__main__':main()
