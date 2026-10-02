#!/usr/bin/env python3
"""Publish complete same-image power comparisons with measured energy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

REPO=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def energy(cases):
    assert len(cases)==70
    assert all(math.isfinite(c['card_energy_j']) and c['card_energy_j']>0 and c['batch_wall_s']>0 for c in cases)
    joules=sum(c['card_energy_j'] for c in cases);seconds=sum(c['batch_wall_s'] for c in cases)
    tokens=sum(c['completion_tokens'] for c in cases)
    assert tokens==124*1024
    return dict(measured_wave_wall_s=seconds,card_energy_j=joules,card_energy_wh=joules/3600,
        mean_card_power_w=joules/seconds,card_j_per_output_token_including_prefill=joules/tokens,
        output_tokens_per_s_including_prefill=tokens/seconds,
        output_tokens_per_s_per_measured_watt_including_prefill=tokens/joules,
        output_tokens_per_wh_including_prefill=tokens*3600/joules,
        completion_tokens=tokens,scope='70 measured waves only, including their prefill+decode and request overhead. Warmup, worker startup and supplementary prefix requests excluded. Card energy, not whole-system electricity.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--additional-campaign',type=Path,action='append',default=[])
    a=p.parse_args();campaign=a.campaign.resolve();root=a.out.resolve();assert not root.exists()
    control=read(campaign/'campaign.json');release=read(REPO/'config/production_image.json')
    assert control['status']=='COMPLETE_POWER_COMPARISON_MEASUREMENTS' and control['success']
    assert control['restored_power_cap_w']==180 and control['restoration']['image_id']==release['image_id']==control['image_id']
    baseline=a.baseline.resolve();roots={180:baseline};controls={}
    for source in [campaign,*[x.resolve() for x in a.additional_campaign]]:
        current=read(source/'campaign.json')
        assert current['status']=='COMPLETE_POWER_COMPARISON_MEASUREMENTS' and current['success']
        assert current['restored_power_cap_w']==180
        assert current['restoration']['image_id']==current['image_id']==release['image_id']
        assert current['policy_sha256']==release['policy_sha256']
        assert set(current['variants'])=={str(w) for w in current['requested_power_caps_w']}
        assert current['variants']
        for key,variant in current['variants'].items():
            watts=int(key)
            assert watts in (150,230,275) and watts not in roots and variant['status']=='COMPLETE'
            roots[watts]=source/f'{watts}w'
        controls[str(source.relative_to(REPO))]=dict(campaign_sha256=sha(source/'campaign.json'),source_commit=current['source_commit'])
    levels=sorted(roots);experiments=[w for w in levels if w!=180]
    title=' / '.join(str(w) for w in levels)+' W'
    profiles={};receipts={}
    original=read(baseline/'source-review-worker/identity.json')
    reference_summary=read(REPO/'benchmarks/results/exl3-review-release-v2/serving-summary.json')
    for watts in levels:
        path=roots[watts]
        summary=reference_summary if watts==180 else read(path/'summary.json')
        identity=read(path/'source-review-worker/identity.json')
        assert identity['Image']==original['Image']==release['image_id']
        assert dict(x.split('=',1) for x in identity['Config']['Env'])==dict(x.split('=',1) for x in original['Config']['Env'])
        assert all(identity['Config'][k]==original['Config'][k] for k in ('Cmd','Entrypoint','Image','Labels'))
        assert summary['source_review']['waves']==70 and summary['source_review']['requests']==124
        assert summary['image_id']==release['image_id'] and summary['policy_sha256']==release['policy_sha256']
        assert summary['source_review']['preemptions']==summary['source_review']['prefill_recompute_excess']==0
        assert summary['source_review']['sampling']==reference_summary['source_review']['sampling']
        assert summary['source_review']['corpus_sha256']==reference_summary['source_review']['corpus_sha256']
        assert summary['source_review']['prompt_hash_verification']['fixture_manifest_sha256']==reference_summary['source_review']['prompt_hash_verification']['fixture_manifest_sha256']
        raw_file=next((path/'source-review').glob('*/results.json'));raw=read(raw_file)
        assert summary['source_review']['raw_results_sha256']==sha(raw_file)
        info=dict(power_cap_w=watts,serving=summary['source_review'],energy=energy(raw['cases']),
                  image_id=summary['image_id'],policy_sha256=summary['policy_sha256'])
        if watts!=180:
            receipt=read(path/'campaign.json');trace=path/'hardware-observations.jsonl'
            assert receipt['status']=='COMPLETE_POWER_SERVING_MATRIX' and receipt['power_cap_w']==watts
            assert sha(trace)==receipt['hardware_observations_sha256']
            samples=[json.loads(line) for line in trace.read_text().splitlines()]
            assert len(samples)>2 and all(x['power_cap_uw']==watts*1_000_000 for x in samples)
            info['hardware']=dict(samples=len(samples),package_temperature_max_c=max(x['package_temperature_c'] for x in samples),
                vram_temperature_max_c=max(x['vram_temperature_c'] for x in samples),observations_sha256=sha(trace),
                scope='2-second sysfs samples over startup, matrix and isolated prefix; thermal samples were not recorded for the earlier 180 W baseline.')
        profiles[str(watts)]=info
        receipts[str(path.relative_to(REPO))+'/source-review-worker/identity.json']=sha(path/'source-review-worker/identity.json')
        receipts[str(raw_file.relative_to(REPO))]=sha(raw_file)
    comparison=dict(status='COMPLETE_MEASURED_POWER_COMPARISON_180W_RESTORED',image_id=release['image_id'],
        policy_sha256=release['policy_sha256'],profiles=profiles,source_receipts_sha256=receipts,
        campaign_sha256=sha(campaign/'campaign.json'),controller_source_commit=control['source_commit'],
        campaigns=controls,
        renderer_sha256=sha(Path(__file__)),limitations=['One full matrix per power cap, each with the existing 3/5 repetitions; not a randomized multi-campaign experiment.',
            'Existing same-image 180 W run reused; temperature traces absent for that baseline.',
            'Output trajectories and speculative acceptance may differ; fixed prompt/request bytes do not guarantee identical generated text.',
            'Additional power caps are measured hardware variants, not a change to the qualified 180 W production policy.',
            'Energy is integrated over measured prefill+decode waves; J/output token is not decode-only energy or whole-system electricity.',
            'Client SSE sampling defines fully overlapped aggregate throughput; it is distinct from native request-weighted decode.'])
    root.mkdir(parents=True)
    write(root/'comparison.json',comparison);shutil.copyfile(campaign/'campaign.json',root/'campaign.json')
    for source in a.additional_campaign:
        shutil.copyfile(source/'campaign.json',root/f'{source.name}-controller-campaign.json')
    for watts in experiments:
        path=roots[watts]
        shutil.copyfile(path/'summary.json',root/f'{watts}w-serving-summary.json')
        shutil.copyfile(path/'campaign.json',root/f'{watts}w-campaign.json')
        shutil.copyfile(path/'hardware-observations.jsonl',root/f'{watts}w-hardware-observations.jsonl')
    def scenario(watts,name):return next(x for x in profiles[str(watts)]['serving']['scenario_results'] if x['name']==name)
    overview='| Power limit | Measured mean card power | Wave time | Output tok/s | Output tok/s per W | Output tok/Wh | Card energy |\n|---:|---:|---:|---:|---:|---:|---:|\n'
    for watts in levels:
        e=profiles[str(watts)]['energy']
        overview+=f"| {watts} W | {e['mean_card_power_w']:.1f} W | {e['measured_wave_wall_s']/60:.2f} min | {e['output_tokens_per_s_including_prefill']:.1f} | {e['output_tokens_per_s_per_measured_watt_including_prefill']:.3f} | {e['output_tokens_per_wh_including_prefill']:,.1f} | {e['card_energy_wh']:.2f} Wh |\n"
    efficient=max(profiles,key=lambda w:profiles[w]['energy']['output_tokens_per_wh_including_prefill'])
    fastest=min(profiles,key=lambda w:profiles[w]['energy']['measured_wave_wall_s'])
    ranked=sorted(profiles,key=lambda w:profiles[w]['energy']['output_tokens_per_wh_including_prefill'],reverse=True)
    runner_up=ranked[1]
    efficiency_lead_percent=(profiles[efficient]['energy']['output_tokens_per_wh_including_prefill']/profiles[runner_up]['energy']['output_tokens_per_wh_including_prefill']-1)*100
    efficiency_note=f'''All efficiency columns cover the same **126,976 output tokens**, including
their prefill and request overhead. `Output tok/s per W` divides whole-wave
throughput by measured mean card power: it equals output tokens/J. `Output
tok/Wh` is output tokens divided by integrated card energy; **higher is
better**. These are not decode-only rates, and do not include whole-PC power.

For this fixed mixed workload, **{efficient} W produced the most tokens per Wh**;
**{fastest} W finished the measured waves fastest**. The rate tables below show
which context and concurrency points benefit from the additional power.

The observed efficiency lead over the next-best setting ({runner_up} W) is
**{efficiency_lead_percent:.1f}%**. Each limit has one full campaign with the fixed
repeats, without randomized order. Small differences do not establish a
general optimum; generated trajectories and MTP acceptance can also differ.

'''
    c1='| C1 input budget | '+' | '.join(f'{w} W prefill / decode' for w in levels)+' |\n|---:|'+('---:|'*len(levels))+'\n'
    names=[('4K','phase-4k-c1'),('16K','phase-16k-c1'),('64K','phase-64k-c1'),('128K','phase-128k-c1'),('200K','full-context-199680')]
    for label,name in names:
        cells=[f"{scenario(w,name)['prefill_tps_median']:,.1f} / {scenario(w,name)['decode_tps_median']:.1f}" for w in levels]
        c1+='| '+label+' | '+' | '.join(cells)+' |\n'
    c4='| Input per request, C4 | '+' | '.join(f'{w} W aggregate decode' for w in levels)+' |\n|---|'+('---:|'*len(levels))+'\n'
    for label,name in [('2K','concurrency-2k-c4'),('4K','concurrency-4k-c4'),('16K','concurrency-16k-c4')]:
        c4+='| '+label+' | '+' | '.join(f"{scenario(w,name)['fully_overlapped_decode_tps']:.1f}" for w in levels)+' |\n'
    relative=str(root.relative_to(REPO))
    heading=f'## Power-limit comparison: {title}\n\n'
    introduction=heading+'''Same immutable EXL3 v2 image, serving arguments, environment values, frozen
prompts and sampling. The existing fresh 180 W matrix is the baseline; the
other measured variants each repeat **20 scenarios / 70 measured waves / 124 requests** plus
an isolated 64K cold/warm resend. All ordinary requests completed without
preemptions. The production service is restored to **180 W**.

These are measured hardware variants; they do not change the qualified
production policy. Power limits are not assumed actual consumption: the table
uses the card energy counter. Measured wave times exclude warmup/startup;
energy includes each measured wave's prefill, decode and request overhead.

'''
    block=introduction+overview+'\n'+efficiency_note+'C1 rates below are median native prefill / request-weighted decode in tok/s.\n\n'+c1+'\nC4 rates are fully overlapped aggregate decode in tok/s, using the same\nsampled-interval definition as the main serving table.\n\n'+c4+f'\n[All 20 load points, acceptance, energy scope and temperatures](docs/EXL3_POWER_COMPARISON.md); [measurement receipts]({relative}/comparison.json).\n'
    readme=REPO/'README.md';text=readme.read_text();start='<!-- BEGIN POWER COMPARISON -->';end='<!-- END POWER COMPARISON -->'
    if start in text:
        assert text.count(start)==text.count(end)==1
        left=text.index(start);right=text.index(end)+len(end);text=text[:left]+start+'\n'+block+end+text[right:]
    else:
        anchor='## Coding and quality results';assert text.count(anchor)==1
        text=text.replace(anchor,start+'\n'+block+end+'\n\n'+anchor)
    readme.write_text(text)
    report='# EXL3 v2 power-limit comparison — 2026-10-02\n\n'+introduction.removeprefix(heading)+overview+'\n'+efficiency_note+c1+'\n'+c4
    report+='\n## Complete matrix\n\nEach cell is median native prefill / request-weighted decode / weighted MTP\nacceptance. Rates are tok/s; C4 request-weighted decode is not aggregate decode.\n\n| Scenario | '+' | '.join(f'{w} W' for w in levels)+' |\n|---|'+('---:|'*len(levels))+'\n'
    for old in profiles['180']['serving']['scenario_results']:
        row=[]
        for w in levels:
            r=scenario(w,old['name']);row.append(f"{r['prefill_tps_median']:.1f} / {r['decode_tps_median']:.1f} / {r['mtp_acceptance']*100:.1f}%")
        report+='| '+old['name']+' | '+' | '.join(row)+' |\n'
    report+='\n## Temperature and attribution\n\n'
    for w in experiments:
        h=profiles[str(w)]['hardware'];report+=f"- {w} W: maximum sampled package {h['package_temperature_max_c']:.1f} °C / VRAM {h['vram_temperature_max_c']:.1f} °C; {h['samples']} hardware samples.\n"
    report+='\nThe 180 W baseline has no comparable thermal trace. One campaign per limit\nwith the fixed repeats is a finite observation, not a randomized confidence\ninterval. Different output trajectories/acceptance and thermal histories can\naffect results. Energy covers only the same 70 measured waves; supplementary\nprefix requests, warmup/startup and whole-PC energy are excluded. No coding\ncampaign or quantization quality study was repeated. Experimental power settings were\naccepted and verified without sudo; each controller restored the original\n180 W cap and healthy production service.\n'
    report+=f"\nImage `{release['image_id']}`; policy `{release['policy_sha256']}` still describes\n180 W production. [Compact comparison](../{relative}/comparison.json),\n[final controller/recovery receipt](../{relative}/campaign.json) and the\nper-power summaries below preserve identities.\n"
    for w in experiments:report+=f'\n- [{w} W summary](../{relative}/{w}w-serving-summary.json) / [{w} W campaign](../{relative}/{w}w-campaign.json).\n'
    (REPO/'docs/EXL3_POWER_COMPARISON.md').write_text(report)
    (root/'README.md').write_text('# Measured EXL3 v2 power comparison\n\nSee [the power comparison report](../../../docs/EXL3_POWER_COMPARISON.md). The 180 W baseline is the prior immutable v2 run. Additional power caps are same-image experiments with the complete matrix and isolated prefix resends. Production remains 180 W.\n')
    write(root/'MANIFEST.json',dict(files_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}))
    print(json.dumps(dict(status=comparison['status'],profiles={w:d['energy'] for w,d in profiles.items()}),indent=2))


if __name__=='__main__':main()
