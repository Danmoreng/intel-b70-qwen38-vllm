"""Compare identical frozen panels without claiming an unpaired promotion."""
import hashlib
import json
from pathlib import Path
import re
import statistics

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'benchmarks/results/dflash2-pro-optimization-v031'


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def native_rate(w):return (w['native']['generation_tokens']-w['native']['completed'])/w['native']['decode_seconds']
def client_rate(w):return (w['native']['generation_tokens']-w['native']['completed'])/w['client_decode_interval_union_s']
def counter(path,name):
    values=[float(line.rsplit(' ',1)[1]) for line in path.read_text().splitlines() if line.startswith(name+'{') or line.startswith(name+' ')]
    assert values,('Missing actual prefill counter',path,name)
    return sum(values)


def admission_evidence(root):
    """Keep actual scheduler observations separate from a reservation model."""
    ledger=read(root/'worker/cache-ledger.json')
    groups=ledger['contexts']['192000']['groups']
    full=[g for g in groups if g['spec']['class']=='FullAttentionSpec']
    bounded=[g for g in groups if g['spec']['class']!='FullAttentionSpec']
    assert len(full)==4 and len(bounded)==14
    fixed=sum(g['resident_pool_blocks_per_request'] for g in bounded)
    counts=[]
    for total in [32768,33280,33288]:
        required=fixed+sum((total+g['spec']['block_size']-1)//g['spec']['block_size'] for g in full)
        counts.append(dict(full_attention_tokens=total,blocks_per_request=required,
            four_request_blocks=4*required,available_blocks=ledger['num_pool_blocks'],
            four_requests_fit_model=4*required<=ledger['num_pool_blocks']))
    path=root/'worker/worker.log'
    observations=[dict(running=int(a),waiting=int(b),kv_usage_percent=float(c))
        for a,b,c in re.findall(r'Running: (\d+) reqs, Waiting: (\d+) reqs, XPU KV cache usage: ([\d.]+)%',path.read_text())]
    assert observations
    return dict(pool_blocks=ledger['num_pool_blocks'],pool_stride_bytes=ledger['pool_stride_bytes'],
        actual_allocated_pool_bytes=ledger['total_device_kv_allocation_bytes'],
        bounded_group_reservation_blocks_per_request=fixed,full_attention_groups=len(full),
        reservation_model=counts,periodic_scheduler_observations=observations,
        largest_periodically_observed_running=max(x['running'] for x in observations),
        observed_running3_waiting1=any(x['running']==3 and x['waiting']==1 for x in observations),
        worker_log=dict(path=str(path.relative_to(REPO)),sha256=digest(path)),
        limits=['Periodic logs sample the scheduler; absence of Running4 is not proof it never occurred.',
            'Reservation model uses actual per-request bounded-group maxima and hypothetical full-attention token counts; it is not an allocator trace or a measured maximum concurrency.',
            'Configured max_num_seqs16 is preserved. Available KV/state capacity still constrains simultaneous long requests.'])


def load(root,kind):
    campaign=read(root/'campaign.json')
    assert campaign['restoration']['image_id']==campaign['production_image']
    assert campaign['restoration']['models_endpoint_healthy']
    assert campaign['power_w']==180 and campaign['power_remeasured'] is False
    if kind=='mtp':
        assert campaign['status']=='COMPLETE_BOUNDED_ABBA_SCREEN'
        arms=[(root/f"arm-{a['index']}-mtp3",a['waves']) for a in campaign['arms'] if a['depth']==3]
        assert len(arms)==2
    else:
        assert campaign['status']=='COMPLETE_MATCHED_PANEL_OBSERVATIONAL_SCREEN'
        assert campaign['max_model_len']==192000 and campaign['block_size']==8 and campaign['depth']==7
        arms=[(root/'worker',campaign['waves'])]
    rows=[];manifest=[dict(path=str((root/'campaign.json').relative_to(REPO)),sha256=digest(root/'campaign.json'))]
    for phase,waves in arms:
        measured={}
        assert len(waves)==9
        for wave in waves:
            p=phase/f"{wave['window']}-c{wave['concurrency']}-{wave['cache_mode']}"/'result.json'
            full=read(p);n=full['native']
            assert n==wave['native']
            assert n['completed']==full['concurrency'] and n['generation_tokens']==full['concurrency']*512
            assert n['prompt_tokens']==full['context_tokens']*full['concurrency'] and n['preemptions']==0
            assert all(r['prompt_ids_verified'] and r['finish_reason']=='length' and r['usage']['completion_tokens']==512 for r in full['responses'])
            if full['cache_mode']=='cold':assert n['cached_tokens']==0
            if full['cache_mode']=='warm':assert n['cached_tokens']>0
            manifest.append(dict(path=str(p.relative_to(REPO)),sha256=digest(p)))
            for name in ['metrics-before.prom','metrics-after.prom']:
                path=p.parent/name;manifest.append(dict(path=str(path.relative_to(REPO)),sha256=digest(path)))
            name='vllm:request_prefill_kv_computed_tokens_sum'
            computed=counter(p.parent/'metrics-after.prom',name)-counter(p.parent/'metrics-before.prom',name)
            assert computed+n['cached_tokens']==n['prompt_tokens'],('Unexpected prefill/recompute accounting',p,computed,n)
            row={k:v for k,v in full.items() if k!='responses'}
            row.update(native_request_weighted_decode_tps=native_rate(full),client_aggregate_decode_tps=client_rate(full),
                mean_client_e2e_s=statistics.mean(r['wall_s'] for r in full['responses']),
                mean_client_ttft_s=statistics.mean(r['ttft_client_s'] for r in full['responses']),
                prefill_computed_tokens=computed,prefill_recompute_excess=0,
                responses=[{k:r[k] for k in ['usage','finish_reason','prompt_ids_verified','output_ids_sha256','output_text_sha256','wall_s','ttft_client_s']} for r in full['responses']])
            if full['cache_mode']!='warmup':measured[full['window'],full['cache_mode']]=row
        assert len(measured)==6
        rows.append(measured)
    return campaign,rows,manifest


def main():
    reference,refs,manifest=load(REPO/'benchmark-results/mtp4-pro-qualification-v031-abba','mtp')
    candidates=[]
    for suffix,projection in [('-full-qkv','full-qkv'),('','kv-only')]:
        root=REPO/('benchmark-results/dflash2-w4-matched-panel-v031'+suffix)
        campaign,arms,raw=load(root,'dflash');assert campaign['context_projection']==projection
        assert campaign['panel']==reference['panel'] and campaign['weights']==reference['weights']
        candidate=arms[0];cells=[];weighted={'cold':0.,'warm':0.}
        for (window,mode),row in candidate.items():
            controls=[r[window,mode] for r in refs]
            assert all(c['context_tokens']==row['context_tokens'] and c['concurrency']==row['concurrency'] for c in controls)
            baseline=statistics.mean(c['native_request_weighted_decode_tps'] for c in controls)
            gain=row['native_request_weighted_decode_tps']/baseline-1
            weighted[mode]+=reference['weights'][window]*gain
            cells.append(dict(window=window,mode=mode,concurrency=row['concurrency'],context_tokens=row['context_tokens'],
                mtp3_native_rates=[c['native_request_weighted_decode_tps'] for c in controls],
                dflash_native_rate=row['native_request_weighted_decode_tps'],relative_native_gain_vs_mean=gain,
                mtp3_client_rates=[c['client_aggregate_decode_tps'] for c in controls],
                dflash_client_rate=row['client_aggregate_decode_tps'],
                client_gains_vs_each_control=[row['client_aggregate_decode_tps']/c['client_aggregate_decode_tps']-1 for c in controls],
                ttft_changes_vs_each_control=[row['mean_client_ttft_s']/c['mean_client_ttft_s']-1 for c in controls],
                e2e_changes_vs_each_control=[row['mean_client_e2e_s']/c['mean_client_e2e_s']-1 for c in controls],
                acceptance_length=row['mean_acceptance_length'],preemptions=row['native']['preemptions']))
        score=statistics.mean(weighted.values())
        c4=[c for c in cells if c['concurrency']==4]
        no_c4_regression=all(min(c['client_gains_vs_each_control'])>=-.05 and max(c['e2e_changes_vs_each_control'])<=.05 and max(c['ttft_changes_vs_each_control'])<=.05 for c in c4)
        admission=admission_evidence(root)
        candidates.append(dict(projection=projection,image=campaign['image'],max_model_len=192000,
            cells=cells,weighted_relative_gains=weighted,combined_equal_cold_warm_gain=score,
            observed_c4_guards_clear_against_both_controls=no_c4_regression,
            warrants_additional_paired_qualification=score>=.05 and no_c4_regression,
            promotion_qualified=False,restoration=campaign['restoration'],admission_evidence=admission))
        receipt=OUT/('panel-'+projection);receipt.mkdir(exist_ok=True)
        (receipt/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
        (receipt/'waves.json').write_text(json.dumps(list(candidate.values()),indent=2)+'\n')
        for method in ['inspect_dflash_setup','w4_numerical_probe','w4_prefill_kv_probe']:
            p=root/(method+'.json')
            if p.exists():
                (receipt/p.name).write_bytes(p.read_bytes());raw.append(dict(path=str(p.relative_to(REPO)),sha256=digest(p)))
        p=root/'worker/cache-ledger.json';ledger=read(p)
        (receipt/'runtime-source-manifest.json').write_text(json.dumps(ledger.pop('sources'),indent=2)+'\n')
        (receipt/'cache-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
        raw.append(admission['worker_log'])
        (receipt/'raw-manifest.json').write_text(json.dumps(raw,indent=2)+'\n')
        manifest+=raw
    result=dict(status='OBSERVATIONAL_MATCHED_FIXTURE_PANEL_NO_PROMOTION',weights=reference['weights'],candidates=candidates,
        raw_manifest=manifest,analysis_source_sha256=digest(Path(__file__)),
        limits=['One DFlash arm per projection, against two earlier MTP3 arms: different time and ordering, no paired confidence interval.',
            'Native C4 request decode seconds include overlap and scheduling while other requests prefill; client aggregate decode and latency are separate guards.',
            '512-output panel rates exclude the first output per request. Initial1024-token screens are a separate trajectory and measurement.',
            '192K capacity is established by separate actual input-plus-output and context-end receipts, not by these32K/64K cells.',
            'Four submitted requests do not imply four simultaneous decode requests. Scheduler observations and a separate reservation model document DFlash KV/state admission pressure.',
            'Stochastic generated prefixes can differ at identical seeds; these comparisons do not establish target-distribution correctness.'])
    (OUT/'panel-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for candidate in candidates:print(candidate['projection'],candidate['weighted_relative_gains'],candidate['combined_equal_cold_warm_gain'],candidate['warrants_additional_paired_qualification'])


if __name__=='__main__':main()
