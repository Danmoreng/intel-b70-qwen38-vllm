#!/usr/bin/env python3
"""Gate sparse hybrid-cache retention at 196K, then verify prefix/state behavior."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
CONTROL=REPO/"benchmarks/experiments/m01-observability/runs/run-20260915-134459/long-context/summary.json"
RESTORE=REPO/"scripts/restore-production.py"
DIAGNOSTICS=REPO/"scripts/run-diagnostics.py"
RETENTION=("--prefix-cache-retention-interval","0")

spec=importlib.util.spec_from_file_location("b70_diagnostics",DIAGNOSTICS)
diag=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(diag)

def save(path: Path,value) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)

def run_prefix_state(inspect: dict,run_dir: Path) -> dict:
    phase=run_dir/"prefix-state"; phase.mkdir()
    arguments=inspect["Config"]["Cmd"].copy(); diag.assert_mtp4(arguments)
    arguments += [*RETENTION,"--enable-logging-iteration-details"]
    command=diag.engine_command(inspect,name="b70-m06-prefix-state",image=inspect["Image"],
                                evidence=phase,arguments=arguments)
    save(phase/"engine-command.json",command)
    process=None; log_handle=None
    try:
        log_handle=(phase/"engine.log").open("w")
        process=subprocess.Popen(command,stdout=log_handle,stderr=subprocess.STDOUT)
        diag.wait_ready(process)
        _,raw=diag.prometheus(); (phase/"startup-metrics.txt").write_text(raw)
        subprocess.run(["python3",str(REPO/"scripts/check-prefix-state.py"),
                        "--output",str(phase/"results.json")],check=True,timeout=1800)
        return json.loads((phase/"results.json").read_text())
    finally:
        diag.stop_container("b70-m06-prefix-state",process)
        if log_handle: log_handle.close()

def assessment(result: dict,control: dict) -> dict:
    scheduler=result["scheduler"]; base=control["scheduler"]
    current_preemptions=scheduler["metric_delta"]["vllm:num_preemptions_total"]
    base_preemptions=base["metric_delta"]["vllm:num_preemptions_total"]
    recompute=scheduler["recomputed_scheduler_tokens"]
    base_recompute=base["recomputed_scheduler_tokens"]
    ttft=result["ttft_s"]; base_ttft=control["ttft_s"]
    gates={
        "prompt_tokens_exact":result["usage"].get("prompt_tokens")==200448,
        "preemptions_reduced":current_preemptions < base_preemptions,
        "scheduler_recompute_reduced":recompute < base_recompute,
        "ttft_at_least_3pct_faster":ttft <= base_ttft*0.97,
        "power_cap_180w":all(row.get("power_cap_uw")==180000000
                              for row in diag.telemetry_rows(Path(result["telemetry_path"]))),
    }
    return {"pass":all(gates.values()),"gates":gates,
            "candidate":{"ttft_s":ttft,"preemptions":current_preemptions,
                         "recomputed_tokens":recompute,"scheduled_tokens":scheduler["scheduled_prefill_tokens"]},
            "control":{"ttft_s":base_ttft,"preemptions":base_preemptions,
                       "recomputed_tokens":base_recompute,"scheduled_tokens":base["scheduled_prefill_tokens"]},
            "ttft_change_pct":100*(ttft/base_ttft-1)}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args(); run_dir=args.run_dir.resolve(); run_dir.mkdir(parents=True,exist_ok=False)
    state={"status":"preflight","started_at_unix":time.time(),"run_dir":str(run_dir),
           "candidate":{"prefix_cache_retention_interval":0,"mtp_depth":4}}
    state_path=run_dir/"state.json"; save(state_path,state)
    lock_path=Path("/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock")
    lock=lock_path.open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    inspect=diag.production_inspect()
    if inspect["Image"]!=diag.EXPECTED_IMAGE_ID: raise RuntimeError("production image changed")
    diag.assert_mtp4(inspect["Config"]["Cmd"])
    if int(diag.POWER_CAP.read_text())!=180000000: raise RuntimeError("power cap is not 180 W")
    diag.ensure_idle(diag.PRODUCTION_URL)
    if not CONTROL.exists(): raise RuntimeError(f"missing frozen control {CONTROL}")
    control=json.loads(CONTROL.read_text()); save(run_dir/"control.json",control)
    save(run_dir/"production-inspect.json",inspect)
    subprocess.run(["python3",str(REPO/"scripts/collect-runtime-manifest.py"),"--container",
                    diag.PRODUCTION_CONTAINER,"--output",str(run_dir/"manifest.json")],
                   check=True,timeout=60,cwd=REPO)
    try:
        state["status"]="stopping-production"; save(state_path,state)
        subprocess.run(["systemctl","--user","stop",diag.PRODUCTION_SERVICE],check=True,timeout=120)
        state["status"]="screen-196k"; save(state_path,state)
        screen=diag.run_long_context(inspect,run_dir,phase_name="screen-196k",
                                     container_name="b70-m06-screen",extra_arguments=RETENTION)
        screen["telemetry_path"]=str(run_dir/"screen-196k/telemetry.jsonl")
        state["screen"]=assessment(screen,control); save(state_path,state)
        if not state["screen"]["pass"]:
            state["status"]="rejected-after-screen"; state["finished_at_unix"]=time.time(); save(state_path,state)
            return 0
        state["status"]="confirm-196k"; save(state_path,state)
        confirm=diag.run_long_context(inspect,run_dir,phase_name="confirm-196k",
                                      container_name="b70-m06-confirm",extra_arguments=RETENTION)
        confirm["telemetry_path"]=str(run_dir/"confirm-196k/telemetry.jsonl")
        state["confirm"]=assessment(confirm,control); save(state_path,state)
        if not state["confirm"]["pass"]:
            state["status"]="rejected-after-confirm"; state["finished_at_unix"]=time.time(); save(state_path,state)
            return 0
        state["status"]="prefix-state"; save(state_path,state)
        state["prefix_state"]=run_prefix_state(inspect,run_dir)
        state["status"]="qualified-for-promotion"; state["finished_at_unix"]=time.time(); save(state_path,state)
        (run_dir/"decision.md").write_text(
            "# M06 preemption decision\n\nStatus: qualified for reviewed production promotion.\n\n"
            "The candidate passed two 196K runs plus cold/warm prefix, state-correctness and MTP gates.\n"
            "No production configuration was changed by this runner.\n")
        return 0
    except BaseException as error:
        state["status"]="failed"; state["error"]=repr(error); state["finished_at_unix"]=time.time(); save(state_path,state)
        raise
    finally:
        subprocess.run(["python3",str(RESTORE)],check=True,timeout=800,
                       env={**os.environ,"B70_RUN_DIR":str(run_dir)})

if __name__=="__main__": raise SystemExit(main())
