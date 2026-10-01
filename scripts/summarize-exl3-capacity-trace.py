#!/usr/bin/env python3
"""Explain traced KV allocation failures without treating logical counters as compute."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

def rows(root):
    for file in sorted(root.glob('trace-*.jsonl*')):
        with (gzip.open(file,'rt') if file.suffix=='.gz' else file.open()) as stream:
            for line in stream:
                yield json.loads(line)

def summarize(root):
    events=sorted(rows(root/'trace'),key=lambda r:r['monotonic'])
    memory_stages=[];pending=defaultdict(list);failures=[];preemptions=[]
    scheduled=Counter();prefill_scheduled=Counter();cache=None;last_allocation={}
    forward_tokens=Counter();forward_prefill=Counter();prompt_lengths={}
    for event in events:
        name=event['event'];request=event.get('request',{});rid=request.get('request_id')
        if rid and request.get('num_prompt_tokens') is not None: prompt_lengths[rid]=request['num_prompt_tokens']
        if name in ('before_weights','after_weights_and_draft','before_draft_weights','after_draft_weights',
                    'before_memory_profile','after_memory_profile','before_graph_capture','after_graph_capture','after_warmup'):
            memory_stages.append({k:v for k,v in event.items() if k not in ('native_libraries',)})
        if name=='cache_initialized': cache=event['cache']
        if name=='before_allocation':
            pending[rid]=[];last_allocation[rid]=event
        if name=='group_allocation_query': pending[event['request_id']].append(event)
        if name=='allocation_rejected':
            queries=pending[rid]
            # Admission and actual allocation can each call every manager. Keep
            # rounds separate rather than double-count their estimates.
            rounds=[];current=[];seen=set()
            for query in queries:
                if query['group'] in seen:
                    rounds.append(current);current=[];seen=set()
                current.append(query);seen.add(query['group'])
            if current: rounds.append(current)
            before=last_allocation.get(rid,{})
            args=before.get('arguments',{})
            calculations=[]
            for group_round in rounds:
                group_demand={str(q['group']):q['requested_blocks'] for q in group_round}
                free=group_round[-1]['free_pool_blocks']
                reserved=args.get('reserved_blocks',0)
                watermark=event['cache']['watermark_blocks'] if 'WAIT' in str(request.get('status')) else None
                calculations.append({'requested_by_group':group_demand,'sum_requested':sum(group_demand.values()),
                                     'free_pool_blocks_at_query':free,'reserved_blocks_argument':reserved,
                                     'raw_deficit_vs_free_pool':max(0,sum(group_demand.values())-free),
                                     'admission_cap':[q['admission_cap'] for q in group_round],
                                     'watermark_note':'Use recorded request enum/state and allocator source for applicable watermark; do not infer from integer status'})
            failures.append({'request_id':rid,'request':request,'cache':event['cache'],'memory':event.get('memory'),
                             'allocation_arguments':args,'query_rounds':calculations,'queries':queries})
        if name=='before_preemption': preemptions.append(event)
        if name=='scheduled_work':
            for request_id,n in event['tokens'].items():
                scheduled[request_id]+=n
                state=event['request_state_before'].get(request_id,{})
                remaining=max(0,(state.get('num_prompt_tokens') or 0)-(state.get('num_computed_tokens') or 0))
                prefill_scheduled[request_id]+=min(n,remaining)
        if name=='target_forward_submission':
            for request_id,n in event['tokens'].items():
                forward_tokens[request_id]+=n
                offset=event['computed_offsets'].get(request_id)
                if offset is not None and request_id in prompt_lengths:
                    forward_prefill[request_id]+=min(n,max(0,prompt_lengths[request_id]-offset))
    native=None
    for name in ('observation.json','operational.json','operational-progress.json'):
        observation=root/name
        if observation.exists():
            native=json.loads(observation.read_text()).get('native')
            if native is not None: break
    return {'schema':1,'trace_rows':len(events),'cache':cache,'memory_stages':memory_stages,
            'preemption_count':len(preemptions),'preemptions':preemptions,
            'allocation_failure_count':len(failures),'allocation_failures':failures,
            'scheduled_tokens_by_request':dict(scheduled),'scheduled_prefill_upper_bound_by_request':dict(prefill_scheduled),
            'submitted_forward_tokens_by_request':dict(forward_tokens),'submitted_prefill_tokens_by_request':dict(forward_prefill),
            'forward_work_note':'Actual Worker.execute_model calls with input offsets, excluding cache-hit token positions. Completion remains asynchronous; no per-kernel timing is inferred.',
            'scheduled_work_limit':'Scheduling can include speculative/rejected/cancelled work. Prefix hits assigned inside schedule can inflate the prefill estimate. This is not a completed-GPU-compute counter.',
            'native_logical_metrics':native,
            'verdict':'No traced preemption' if not preemptions else 'Preemption recorded; inspect per-group allocator rounds and request state',
            'unmeasured':['oneDNN private workspace allocation','exact graph-pool ownership','completed GPU kernel token work']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.case),indent=2)+'\n')
