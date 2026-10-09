"""Export actual eager gates and preserve the failed capture/reference attempts."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
RAW=REPO/'benchmark-results/dflash2-pro-diagnostics-v031'
OUT=REPO/'benchmarks/results/dflash2-pro-optimization-v031/actual-eager'


def main():
    root=RAW/'a1-eager-branches-20261010-attempt3'
    campaign=json.loads((root/'campaign.json').read_text())
    assert campaign['status']=='PASS_ACTUAL_EAGER_BRANCH_REPLAY'
    assert campaign['restoration']['image_id']==campaign['production_image']
    assert campaign['restoration']['models_endpoint_healthy']
    OUT.mkdir(exist_ok=True)
    manifest=[]
    for source,target in [
        ('campaign.json','campaign.json'),
        ('worker/branch-capture/replay.json','branch-replay.json'),
        ('proposal-audit.json','proposal-audit.json'),
        ('worker/epsilon-probe.json','epsilon-probe.json'),
        ('worker/rejection-greedy-graph.json','rejection-greedy-graph.json'),
        ('worker/identity.json','worker-attestation.json'),
        ('worker/draft-identity.json','draft-identity.json'),
    ]:
        p=root/source
        manifest.append(dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
        (OUT/target).write_bytes(p.read_bytes())
    ledger=json.loads((root/'worker/cache-ledger.json').read_text())
    (OUT/'runtime-source-manifest.json').write_text(json.dumps(ledger.pop('sources'),indent=2)+'\n')
    (OUT/'cache-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    for path in sorted((root/'worker/branch-capture').glob('round-*.pt')):
        manifest.append(dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    (OUT/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    failures=[]
    for name,explanation in [
        ('a1-eager-branches-20261010','Capture counted prefill calls; a4096-row feature FC input was compared with eight captured feature rows. Decode-only selection fixes the capture geometry.'),
        ('a1-eager-branches-20261010-attempt2','Ideal FP64 RoPE rotation exceeded frozen bounds. The passing diagnostic splits cache-theta validation and rotation using actual stored FP16 coefficients at the same bounds; ideal FP64 rotation errors remain explicitly reported. Absolute sample positions were also corrected to query-row indices before a fresh capture.'),
    ]:
        p=RAW/name/'campaign.json';c=json.loads(p.read_text())
        assert c['status']=='FAILED' and c['restoration']['models_endpoint_healthy']
        failures.append(dict(campaign=c,explanation=explanation,raw_path=str(p.relative_to(REPO)),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),replay_log=(p.parent/'replay.log').read_text()))
    (OUT/'failed-attempts.json').write_text(json.dumps(failures,indent=2)+'\n')
    print('PUBLISHED_ACTUAL_EAGER_GATES_WITH_FAILED_ATTEMPTS')


if __name__=='__main__':main()
