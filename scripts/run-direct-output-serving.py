#!/usr/bin/env python3
"""Run cold 65K ABBA serving and a gated 196K direct-output qualification."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
CONTROL_IMAGE="sha256:3f20b0cf493fe0904a7efd0ca310067790e901bc5d60203340c411e57c25010a"
CANDIDATE_IMAGE="local/qwen38-b70-vllm:q128-direct-output-candidate"
CANDIDATE_ID="sha256:391003ca49afdadee297079a4f24a861c896004332d032ef3fa74d5e1f7cdea7"
spec=importlib.util.spec_from_file_location("b70_diagnostics",REPO/"scripts/run-diagnostics.py")
diag=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(diag)

def save(path: Path,value) -> None:
    path.write_text(json.dumps(value,indent=2)+"\n")

def run_arm(inspect: dict,run_dir: Path,index: int,label: str,image: str) -> dict:
    phase=run_dir/f"arm-{index}-{label}"; phase.mkdir()
    arguments=inspect["Config"]["Cmd"].copy(); diag.assert_mtp4(arguments)
    command=diag.engine_command(inspect,name="b70-direct-serving-arm",image=image,
                                evidence=phase,arguments=arguments)
    save(phase/"engine-command.json",command)
    process=None; log_handle=None
    try:
        log_handle=(phase/"engine.log").open("w")
        process=subprocess.Popen(command,stdout=log_handle,stderr=subprocess.STDOUT)
        diag.wait_ready(process)
        subprocess.run(["python3",str(REPO/"scripts/benchmark.py"),"--root",diag.EXPERIMENT_URL,
                        "--contexts","65536","--output-tokens","32","--repeats","1",
                        "--timeout","1800","--output-dir",str(phase/"benchmark")],
                       check=True,timeout=2200,cwd=REPO)
        row=json.loads((phase/"benchmark/results.json").read_text())["rows"][0]
        row.update({"arm":index,"label":label,"image":image})
        save(phase/"summary.json",row); return row
    finally:
        diag.stop_container("b70-direct-serving-arm",process)
        if log_handle: log_handle.close()

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args(); run_dir=args.run_dir.resolve(); run_dir.mkdir(parents=True,exist_ok=False)
    state={"status":"preflight","started_at_unix":time.time(),"run_dir":str(run_dir),"arms":[]}
    state_path=run_dir/"state.json"; save(state_path,state)
    lock_path=Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock=lock_path.open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    inspect=diag.production_inspect()
    if inspect["Image"]!=CONTROL_IMAGE: raise RuntimeError("production image changed")
    actual=subprocess.check_output(["docker","image","inspect",CANDIDATE_IMAGE,"--format","{{.Id}}"],text=True).strip()
    if actual!=CANDIDATE_ID: raise RuntimeError(f"candidate image changed: {actual}")
    if int(diag.POWER_CAP.read_text())!=180000000: raise RuntimeError("power cap is not 180 W")
    diag.ensure_idle(diag.PRODUCTION_URL); save(run_dir/"production-inspect.json",inspect)
    try:
        state["status"]="stopping-production"; save(state_path,state)
        subprocess.run(["systemctl","--user","stop",diag.PRODUCTION_SERVICE],check=True,timeout=120)
        order=[("control",CONTROL_IMAGE),("candidate",CANDIDATE_IMAGE),
               ("candidate",CANDIDATE_IMAGE),("control",CONTROL_IMAGE)]
        for index,(label,image) in enumerate(order):
            state["status"]=f"arm-{index}-{label}"; save(state_path,state)
            state["arms"].append(run_arm(inspect,run_dir,index,label,image)); save(state_path,state)
        grouped={label:[row for row in state["arms"] if row["label"]==label]
                 for label in ("control","candidate")}
        def median(label,key): return statistics.median(row[key] for row in grouped[label])
        assessment={
            "control_prefill_s":median("control","prefill_seconds"),
            "candidate_prefill_s":median("candidate","prefill_seconds"),
            "control_ttft_s":median("control","ttft_seconds"),
            "candidate_ttft_s":median("candidate","ttft_seconds"),
            "control_decode_tps":median("control","decode_tokens_per_second"),
            "candidate_decode_tps":median("candidate","decode_tokens_per_second"),
        }
        assessment["prefill_speedup_pct"]=100*(assessment["control_prefill_s"]/assessment["candidate_prefill_s"]-1)
        assessment["ttft_speedup_pct"]=100*(assessment["control_ttft_s"]/assessment["candidate_ttft_s"]-1)
        assessment["decode_change_pct"]=100*(assessment["candidate_decode_tps"]/assessment["control_decode_tps"]-1)
        assessment["pass_65k"]=(assessment["prefill_speedup_pct"]>=1 and assessment["ttft_speedup_pct"]>=1
                                and assessment["decode_change_pct"]>=-5)
        state["assessment_65k"]=assessment; save(state_path,state)
        if not assessment["pass_65k"]:
            state["status"]="rejected-after-65k"; state["finished_at_unix"]=time.time(); save(state_path,state); return 0
        state["status"]="candidate-196k"; save(state_path,state)
        result=diag.run_long_context(inspect,run_dir,phase_name="candidate-196k",
                                     container_name="b70-direct-serving-196k",image=CANDIDATE_IMAGE)
        control=json.loads((REPO/"benchmarks/experiments/m01-observability/runs/run-20260915-134459/long-context/summary.json").read_text())
        long_assessment={"candidate_ttft_s":result["ttft_s"],"control_ttft_s":control["ttft_s"],
                         "candidate_preemptions":result["scheduler"]["metric_delta"]["vllm:num_preemptions_total"],
                         "control_preemptions":control["scheduler"]["metric_delta"]["vllm:num_preemptions_total"],
                         "candidate_recomputed_tokens":result["scheduler"]["recomputed_scheduler_tokens"],
                         "control_recomputed_tokens":control["scheduler"]["recomputed_scheduler_tokens"]}
        long_assessment["ttft_speedup_pct"]=100*(control["ttft_s"]/result["ttft_s"]-1)
        long_assessment["pass"]=(long_assessment["ttft_speedup_pct"]>=1 and
                                  long_assessment["candidate_preemptions"]<=long_assessment["control_preemptions"] and
                                  long_assessment["candidate_recomputed_tokens"]<=long_assessment["control_recomputed_tokens"])
        state["assessment_196k"]=long_assessment
        state["status"]="qualified-for-promotion" if long_assessment["pass"] else "rejected-after-196k"
        state["finished_at_unix"]=time.time(); save(state_path,state); return 0
    except BaseException as error:
        state["status"]="failed"; state["error"]=repr(error); state["finished_at_unix"]=time.time(); save(state_path,state); raise
    finally:
        subprocess.run(["python3",str(REPO/"scripts/restore-production.py")],check=True,timeout=800,
                       env={**os.environ,"B70_RUN_DIR":str(run_dir)})

if __name__=="__main__": raise SystemExit(main())
