#!/usr/bin/env python3
"""Verify sparse hybrid-cache prefix reuse, deterministic state and MTP use."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

METRICS = {
    "hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "drafted": ("vllm:spec_decode_num_draft_tokens_total",),
    "accepted": ("vllm:spec_decode_num_accepted_tokens_total",),
    "preemptions": ("vllm:num_preemptions_total",),
}

def snapshot(root: str) -> dict[str,float]:
    text=urllib.request.urlopen(root+"/metrics",timeout=30).read().decode()
    result={}
    for logical,candidates in METRICS.items():
        for candidate in candidates:
            values=[float(line.split()[-1]) for line in text.splitlines()
                    if line.startswith(candidate+"{") or line.startswith(candidate+" ")]
            if values:
                result[logical]=sum(values); break
        else: raise RuntimeError(f"missing metric {logical}: {candidates}")
    return result

def request(root: str, model: str, messages: list[dict]) -> dict:
    before=snapshot(root)
    body={"model":model,"messages":messages,"max_tokens":32,"temperature":0.0,
          "stream":True,"stream_options":{"include_usage":True},
          "reasoning_effort":"none","thinking_token_budget":0,
          "chat_template_kwargs":{"enable_thinking":False,"preserve_thinking":True}}
    req=urllib.request.Request(root+"/v1/chat/completions",data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    started=time.monotonic(); first=None; parts=[]; usage={}
    with urllib.request.urlopen(req,timeout=3600) as response:
        for raw in response:
            line=raw.decode(errors="replace").strip()
            if not line.startswith("data:"): continue
            payload=line[5:].strip()
            if payload=="[DONE]": break
            event=json.loads(payload); usage=event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta=choice.get("delta") or {}
                part=delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                if part: first=first or time.monotonic(); parts.append(part)
    if first is None: raise RuntimeError("request produced no response token")
    after=snapshot(root)
    return {"ttft_s":first-started,"total_s":time.monotonic()-started,
            "usage":usage,"answer":"".join(parts).strip(),
            "metric_delta":{key:after[key]-before[key] for key in before}}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="http://127.0.0.1:18087")
    parser.add_argument("--model",default="Qwen3.8-27B")
    parser.add_argument("--records",type=int,default=800)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    markers={113:"ALPHA-731",args.records//2:"BETA-428",args.records-71:"GAMMA-905"}
    rows=[f"Record {i:04d}: subsystem datum {markers.get(i,f'ITEM-{i:04d}')}; retain this exact mapping."
          for i in range(args.records)]
    system=f"Run nonce {uuid.uuid4().hex}. Answer only from this document.\n"+"\n".join(rows)
    cases=[("Return only the code at record 0113.","ALPHA-731"),
           (f"Return only the code at record {args.records//2:04d}.","BETA-428"),
           (f"Return only the code at record {args.records-71:04d}.","GAMMA-905")]
    records=[]
    for question,expected in cases:
        row=request(args.root,args.model,[{"role":"system","content":system},
                                           {"role":"user","content":question}])
        row["expected"]=expected; row["correct"]=expected in row["answer"]
        if not row["correct"]: raise RuntimeError(f"state mismatch: {row}")
        records.append(row); print(json.dumps(row),flush=True)
    if records[0]["metric_delta"]["hits"] != 0: raise RuntimeError("cold request hit cache")
    for row in records[1:]:
        if row["metric_delta"]["hits"] <= 0: raise RuntimeError("warm request missed prefix cache")
        if row["metric_delta"]["drafted"] <= 0: raise RuntimeError("MTP was not exercised")
    if any(row["metric_delta"]["preemptions"] for row in records):
        raise RuntimeError("prefix/state check triggered a preemption")
    result={"status":"PASS","requests":records}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
