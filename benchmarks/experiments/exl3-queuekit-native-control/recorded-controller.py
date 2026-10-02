import fcntl,json,pathlib,subprocess,sys,time
repo=pathlib.Path('/home/sebastian/LocalLLM/intel-b70-qwen38-vllm-onednn')
sys.path.insert(0,str(repo/'scripts'))
from exl3_candidate_worker import Worker,sha,BASE
root=repo/'benchmark-results/exl3-production-v1/queuekit-native-control-20261002'
with (repo.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 final=json.loads((repo/'benchmark-results/exl3-production-v1/readme-20261002/campaign.json').read_text())
 assert final['status']=='COMPLETE_FINAL_README_MEASUREMENTS_REQUIRES_RELEASE_REVIEW',final['status']
 assert not root.exists()
 image=final['image_receipt']['image_id'];policy=repo/'config/experiments/exl3-migration/final-candidate-policy.json'
 assert sha(policy)==final['image_receipt']['policy_sha256']
 root.mkdir(parents=True)
 state={'status':'RUNNING_CONTROL_NOT_RELEASE_PROFILE','image_id':image,'started_unix':time.time(),
        'single_override':{'EXL3_GUARDED_ATTN':'0'},'reason':'Same frozen QueueKit task; investigate optimized arm 7/8 task outcome without changing its output or retrying its seed.'}
 def save():(root/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
 save();w=Worker(image,root/'worker',name='b70-exl3-queuekit-native-control',env=state['single_override'])
 try:
  w.start()
  argv=[sys.executable,str(repo/'scripts/run-coding-benchmark.py'),'--base',BASE,'--container',w.name,
        '--policy-sha256-file',str(policy.with_suffix('.sha256')),'--output-root',str(root/'coding')]
  state['command']=argv;save()
  with (root/'coding.log').open('w') as log:subprocess.run(argv,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
  result=json.loads((root/'coding/summary.json').read_text())
  state.update(status='COMPLETE_NATIVE_ATTENTION_CONTROL_REQUIRES_REVIEW',tasks_passed=sum(x['acceptance']['passed'] for x in result['task_results']),tasks_total=len(result['task_results']),summary_sha256=sha(root/'coding/summary.json'))
 except BaseException as exc:
  state.update(status='FAILED_CONTROL',error=repr(exc));raise
 finally:
  w.stop();state['finished_unix']=time.time();save()
