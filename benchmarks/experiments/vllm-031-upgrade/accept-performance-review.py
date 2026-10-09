#!/usr/bin/env python3
"""Record an explicit engineering review of measured v0.31 tradeoffs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed-measurements-sha256', required=True)
    a = p.parse_args()
    out = REPO / 'benchmarks/results/vllm-031-upgrade'
    measurement_path = out / 'performance-review-measurements.json'
    assert sha(measurement_path) == a.reviewed_measurements_sha256
    measurement = json.loads(measurement_path.read_text())
    assert measurement['status'] == 'COMPLETE_64K_MEASUREMENTS_REQUIRE_ENGINEERING_REVIEW'
    for receipt in measurement['receipts'].values():
        assert sha(REPO / receipt['path']) == receipt['sha256']
    decision_path = out / 'decision.json'
    decision = json.loads(decision_path.read_text())
    assert decision['status'] == 'REQUIRES_PERFORMANCE_REVIEW'
    assert decision['performance_review_triggers'] == ['phase-64k-c1']
    assert decision['image_id'] == measurement['image_id']
    comparison_path = out / 'benchmark-comparison.json'
    comparison = json.loads(comparison_path.read_text())
    assert [x['name'] for x in comparison['scenarios'] if x['review_trigger']] == decision['performance_review_triggers']
    assessment = json.loads((REPO / decision['gate_assessment']['path']).read_text())
    assert assessment['status'] == 'PASS_EXACT_IMAGE_RELEASE_GATES'
    assert sha(REPO / decision['gate_assessment']['path']) == decision['gate_assessment']['sha256']
    review_path = out / 'performance-review.json'
    before_path = out / 'decision-before-performance-review.json'
    assert not review_path.exists() and not before_path.exists()
    shutil.copyfile(decision_path, before_path)
    review = dict(
        status='PASS_REVIEWED_PERFORMANCE_TRADEOFF', image_id=decision['image_id'],
        reviewed_triggers=decision['performance_review_triggers'], baseline_regenerated=False,
        receipts=dict(measurements=dict(path=str(measurement_path.relative_to(REPO)), sha256=sha(measurement_path)),
                      full_comparison=dict(path=str(comparison_path.relative_to(REPO)), sha256=sha(comparison_path)),
                      original_decision=dict(path=str(before_path.relative_to(REPO)), sha256=sha(before_path))),
        findings=[
            '64K native request-weighted decode median is 48.294 tok/s versus stored v2 52.673 (-8.313%). A fresh new-image repeat gives 48.310 (-8.284%). This is a reproducible regression on these frozen tasks and remains in the primary table.',
            'All three repeated outputs exactly match the new full campaign; MTP acceptance is unchanged between new campaigns (56.050%, stored v2 60.703%). The old outputs differ. No claim that generated text remains identical across versions.',
            'Median same-window round-equivalent costs are 54.564 ms stored versus 54.239 ms new full and approximately 54.263 ms repeat. Lower acceptance with similar round cost is consistent with changed generated trajectories; this does not establish the exact upstream cause or rule out every acceptance-related issue.',
            '64K end-to-end median is about 2.6% longer, while TTFT improves about 0.6%. Fully overlapped aggregate decode falls about 4.6%. These distinct measures are retained rather than substituting one for the native decode regression.',
            'The other 19 frozen load points have no native-decode decrease or TTFT increase crossing the 5% review trigger. Decode changes are mixed; 4K C1 +1.63%, 128K C1 +5.16%, 200K -1.54%. No general speedup claim.',
            'Qualification accepts this explicit bounded workload tradeoff for the authorized runtime upgrade: exact-image original-tolerance numerical/kernel gates, independent FP32 comparisons, matched-state replays, serving/media/restart gates and 262144-token capacity pass. No context, dtype, MTP3 or quality tolerances were reduced to improve throughput. Saved qualified v2 remains the immediate rollback.'
        ],
        scope='Engineering release decision after inspecting hashed full and new-only repeat evidence. Not statistical equivalence, universal accuracy or a speed guarantee. The historical old baseline was not rerun; only the new image used the GPU.')
    review_path.write_text(json.dumps(review, indent=2) + '\n')
    decision.update(status='QUALIFIED_FOR_PRODUCTION_PROMOTION',
                    performance_review=dict(path=str(review_path.relative_to(REPO)), sha256=sha(review_path)),
                    rationale=decision['rationale'] + ' Reviewed reproducible 64K decode regression (-8.3%) is accepted explicitly; remaining measured load points are near the stored baseline. No universal speedup or bitidentical-generation claim; v2 rollback retained.')
    decision_path.write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(dict(status=decision['status'], performance_review_sha256=sha(review_path)), indent=2))


if __name__ == '__main__':
    main()
