"""Close the bounded work order only when no tested arm warrants promotion work."""
import hashlib
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'benchmarks/results/dflash2-pro-optimization-v031'


def read(path):return json.loads(path.read_text())


def main():
    panel=read(OUT/'panel-comparison.json')
    assert panel['status']=='OBSERVATIONAL_MATCHED_FIXTURE_PANEL_NO_PROMOTION'
    assert all(not c['warrants_additional_paired_qualification'] for c in panel['candidates']), 'Promising candidate needs paired qualification; do not close the goal'
    mtp=read(OUT/'mtp4-bounded/analysis.json')
    assert not mtp['clears_point_gain_and_descriptive_interval']
    grouping=read(OUT/'grouping-matched96.json');assert grouping['deterministic_outputs_all_equal']
    assert read(OUT/'actual-eager/campaign.json')['status']=='PASS_ACTUAL_EAGER_BRANCH_REPLAY'
    caps=[]
    for name in ['pro-w4-group4-k7','pro-w4-group4-kv-only-k7']:
        cap=read(OUT/name/'exact-context.json');end=read(OUT/name/'context-end-state.json')
        assert cap['usage']['total_tokens']==192000 and cap['unexpected_recompute_tokens']==0
        assert end['sampling_usage']['total_tokens']==192000 and end['custom_stop_usage']['completion_tokens']==1
        caps.append(dict(candidate=name,usage=cap['usage'],context_end=end))
    spec=importlib.util.spec_from_file_location('restore',REPO/'scripts/run-exl3-review-micro.py')
    restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
    image=read(REPO/'config/experiments/vllm-031-upgrade/production_image.json')['image_id']
    restoration=restore.wait_for_qualified_service(image,timeout=900)
    decision=dict(status='COMPLETE_BOUNDED_WORK_ORDER_NO_SERVING_PROMOTION',
        scope='BEST_WITHIN_TESTED_CANDIDATES_NOT_GLOBAL_OPTIMUM',
        production_choice='Retain qualified FP16 MTP3, FP8 KV,262144 total tokens,max_num_seqs16 and180W.',
        production_restoration=restoration,production_defaults_modified=False,
        preferred_dflash_total_context=192000,conditionally_acceptable_total_context=166000,
        context_policy='User permitted about166K only for a clear speed advantage; actual192K passed, so a166K fallback capacity run is unnecessary.',
        mtp4=dict(decision='NO_PROMOTION_FOR_PREDECLARED_MIX',weighted_gain=mtp['intervals']['combined_equal_cold_warm_screen'],
            reason='Long C1 is faster, but the35/35/30 mix misses5% gain and warm prose/C4 throughput/latency guards. No further depth matrix.',
            long_c1_scope='Potential specialist profile, not a promoted general serving default.'),
        dflash_candidates=panel['candidates'],actual_capacity_gates=caps,
        concurrency_scope='max_num_seqs16 is unchanged, but long-request KV/state admission is constrained by available pool capacity. Actual periodic scheduler observations and an explicitly hypothetical reservation model are included per DFlash candidate; C4 means four submitted requests, not a guaranteed four-request simultaneous decode batch.',
        grouping=dict(decision='192K_CAPACITY_ENABLED_SPEED_NEUTRALITY_NOT_PROVEN',
            deterministic_outputs_all_equal=True,greedy_controls=grouping['greedy_controls'],
            stochastic_cases=grouping['stochastic_cases'],
            reason='Memory padding is reduced. Small negative greedy rate differences and stochastic variance do not establish zero overhead or an isolated speed gain.'),
        static_b8_v4=dict(decision='CONDITIONAL_DEVELOPMENT_NOT_TRIGGERED_BY_LONG_C1_PROFILE',implemented=False,
            reason='Measured prefix truncation needs about14–17% round-time savings. C1 EXL3 DPAS retains MB8 for five/eight target rows, M04 retains tile8, and the eight-row draft backbone remains. The profile does not supply an evidence-backed long-C1 saving of that size.',
            limits='This is a development stop decision, not proof B8/V4 is universally slower. C4 MB24 versusMB32,64K split differences, shortened draft-head work and potentially smaller target speculative-state reservations could help. B8/V4 and its C4 admission behavior were not implemented or measured.'),
        conditional_diagnostics=dict(fp16_draft_kv='No relevant same-stored-KV attention error triggered a separate dtype experiment.',
            original_target_teacher_forcing='Not measured against original unquantized target weights; EXL3 dequantization would not isolate original quantization loss.',
            thinking_task_mix='No causal task/thinking acceptance claim or extra task matrix from these frozen fixtures.'),
        correctness_scope=['Independent actual eager feature/FC/KV/norm/conv/selector/RoPE/paged-KV replay and small-epsilon probes pass their explicitly scoped frozen bounds.',
            'Ideal FP64 full RoPE rotation exceeds original bounds at32K. Cache-theta and kernel-with-actual-FP16-coefficients gates are separate; ideal errors and failed attempts remain published.',
            'Actual greedy W4 acceptance/residual/bonus decisions pass scalar Philox/Murmur/Gumbel replay. Older probabilistic captures have acceptance/q coverage but lack attested Gumbel-mode winner coverage.',
            'Graph proposal/sampler and synthetic paged-attention gates are separate from eager branches. Actual graph-internal model branch equivalence and broad semantic qualification are not claimed.',
            'A changed draft is not promoted solely by finite outputs, acceptance, capacity or these observational rates.'],
        power=dict(selected_cap_w=180,remeasured=False,historical_comparisons_retained=True),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(decision['status'],restoration['image_id'])


if __name__=='__main__':main()
