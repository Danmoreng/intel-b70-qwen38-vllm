#!/usr/bin/env python3
"""Run the isolated b70_ops direct-output correctness and timing gate."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]
IMAGE="local/qwen38-b70-vllm:q128-196k-20260914"
IMAGE_ID="sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a"
CONTAINER="b70-direct-output-probe"

def save(path: Path,value) -> None:
    path.write_text(json.dumps(value,indent=2)+"\n")

def idle() -> None:
    raw=urllib.request.urlopen("http://127.0.0.1:8081/metrics",timeout=10).read().decode()
    for metric in ("vllm:num_requests_running","vllm:num_requests_waiting"):
        values=[]
        for line in raw.splitlines():
            if line.startswith(metric+"{") or line.startswith(metric+" "):
                values.append(float(line.split()[-1]))
        if not values or sum(values)!=0: raise RuntimeError(f"production busy: {metric}={values}")

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args(); run_dir=args.run_dir.resolve(); run_dir.mkdir(parents=True,exist_ok=False)
    state={"status":"preflight","started_at_unix":time.time(),"run_dir":str(run_dir)}
    save(run_dir/"state.json",state)
    lock_path=Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock=lock_path.open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if subprocess.check_output(["docker","image","inspect",IMAGE,"--format","{{.Id}}"],text=True).strip()!=IMAGE_ID:
        raise RuntimeError("runtime image identity changed")
    cap=next(Path("/sys/bus/pci/devices/0000:03:00.0/hwmon").glob("*/power1_cap"))
    if int(cap.read_text())!=180000000: raise RuntimeError("power cap is not 180 W")
    idle()
    command=["docker","run","--rm","--name",CONTAINER,"--network","none","--device","/dev/dri",
             "--group-add",str(Path("/dev/dri/renderD128").stat().st_gid),
             "-e","ZE_AFFINITY_MASK=0","-e","ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
             "-v",f"{run_dir}:/work","-v",f"{REPO/'b70_ops/probe.py'}:/probe.py:ro",
             "-w","/tmp","--entrypoint","python",IMAGE,"/probe.py"]
    save(run_dir/"command.json",command)
    for source,name in ((REPO/"b70_ops/build/b70_ops.so","b70_ops.so"),
                        (REPO/"docker/q128/tiles.so","deployed-tiles.so")):
        destination=run_dir/name
        destination.write_bytes(source.read_bytes())
    try:
        state["status"]="stopping-production"; save(run_dir/"state.json",state)
        subprocess.run(["systemctl","--user","stop","qwen38.service"],check=True,timeout=120)
        state["status"]="operator-probe"; save(run_dir/"state.json",state)
        with (run_dir/"probe.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        results=json.loads((run_dir/"results.json").read_text())
        speedups=[row["direct_speedup_pct"] for row in results["rows"]]
        result={"all_correct":all(row["finite"] and row["allclose"] and row["returned_same_storage"]
                                  for row in json.loads((run_dir/"correctness.json").read_text())),
                "median_speedup_pct":sorted(speedups)[len(speedups)//2],
                "worst_shape_speedup_pct":min(speedups),
                "no_shape_regresses_beyond_1pct":all(value>=-1 for value in speedups)}
        result["pass"]=result["all_correct"] and result["no_shape_regresses_beyond_1pct"] and result["median_speedup_pct"]>=0.5
        state["assessment"]=result; state["status"]="completed"; state["finished_at_unix"]=time.time()
        save(run_dir/"state.json",state)
        return 0
    except BaseException as error:
        state["status"]="failed"; state["error"]=repr(error); state["finished_at_unix"]=time.time()
        save(run_dir/"state.json",state); raise
    finally:
        subprocess.run(["docker","stop","-t","30",CONTAINER],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=45)
        subprocess.run(["python3",str(REPO/"scripts/restore-production.py")],check=True,timeout=800,
                       env={**os.environ,"B70_RUN_DIR":str(run_dir)})

if __name__=="__main__": raise SystemExit(main())
