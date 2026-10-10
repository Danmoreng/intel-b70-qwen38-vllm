"""Sequential gates after bounded D64; stops on any failed prerequisite."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
root=REPO/'benchmark-results/b70-stack-20261010';image='sha256:b772d37937d188bc8492e2e796eafc46668290e68d356f648257e99b06967048'
print('WAIT_BOUNDED_D64_AND_RESTORATION',flush=True)
while True:
 try:os.kill(a.wait_pid,0)
 except ProcessLookupError:break
 time.sleep(15)
assert json.loads((root/'D64-bounded-r1/campaign.json').read_text())['status']=='COMPLETE_BOUNDED_D64_CAPTURE_COMPARISON_PENDING'
public=REPO/'benchmarks/results/b70-stack-20261010'
cmd=['docker','run','--rm','--network','none','--memory','8g','-v',str(HERE.parent)+':/work/benchmarks/experiments:ro','-v',str(root/'D64-bounded-r1')+':/results:ro','-v',str(public)+':/public','--entrypoint','python',image,'-u','/work/benchmarks/experiments/b70-stack-20261010/compare_d64.py','--root','/results','--out','/public/D64-same-prefix-review.json']
subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(HERE/'run_quality.py'),'--image',image,'--out',str(root/'E01-quality')],check=True)
assert json.loads((root/'E01-quality/campaign.json').read_text())['status']=='PASS_NEW_IMAGE_QUALITY_AND_MATCHED_STATE'
# Additional blocks retain all original measurements; no outlier replacement.
screen=root/'E01-confirmation-protected';screen.mkdir(exist_ok=False)
# The serving controller owns its root creation, so remove the empty reservation.
screen.rmdir()
cmd=[sys.executable,'-u',str(HERE/'serving_screen.py'),'--image',image,'--onednn-receipt',str(root/'E01-oneDNN-build/library-receipt.json'),'--blocks','3','--protected','--out',str(screen)]
worker=subprocess.Popen(cmd,cwd=REPO)
while not screen.exists() and worker.poll() is None:time.sleep(.25)
monitor=None
if worker.poll() is None:
 monitor=subprocess.Popen([sys.executable,str(HERE/'monitor_comparability.py'),'--controller-pid',str(worker.pid),'--out',str(screen/'comparability.jsonl')],cwd=REPO)
code=worker.wait()
if monitor is not None:monitor.wait(timeout=45)
assert code==0,('Confirmation failed',code)
subprocess.run([sys.executable,str(HERE/'analyze_screen.py'),'--root',str(root/'E01-serving-screen'),'--additional',str(screen),'--out',str(public/'E01-confirmation-six-blocks.json')],check=True)
print('MAINTENANCE_CONFIRMATION_COMPLETE_REVIEW_REQUIRED_NOT_RELEASE_APPROVAL',flush=True)
