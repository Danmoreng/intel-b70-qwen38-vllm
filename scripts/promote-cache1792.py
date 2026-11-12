"""Run the reviewed deployment with the shared GPU lock and daily bound."""
import datetime as dt,fcntl,importlib.util,signal,sys
from pathlib import Path
from zoneinfo import ZoneInfo

def main():
    repo=Path(__file__).resolve().parents[1]
    now=dt.datetime.now(ZoneInfo('Europe/Berlin'));limit=now.replace(hour=22,minute=35,second=0,microsecond=0)
    assert (limit-now).total_seconds()>30*60
    def interrupt(sig,frame):raise InterruptedError(str(sig))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM]:signal.signal(sig,interrupt)
    spec=importlib.util.spec_from_file_location('reviewed_promotion',repo/'scripts/promote-exl3-review-release.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with Path('/tmp/b70-v033-qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.alarm(1800)
        try:
            sys.argv=[str(repo/'scripts/promote-exl3-review-release.py'),'--release-dir',str(repo/'config/experiments/cache1792-v7'),'--decision',str(repo/'benchmarks/results/mtp-cache-1792-20261112/decision.json'),'--out',str(repo/'benchmarks/results/mtp-cache-1792-20261112/deployment'),'--rollback-release',str(repo/'config/releases/exl3-v6'),'--production-tag','local/b70-qwen38-vllm:production-exl3-v7','--runtime-dir',str(repo/'config/experiments/cache1792-v7/runtime')]
            module.main()
        finally:signal.alarm(0)

if __name__=='__main__':main()
