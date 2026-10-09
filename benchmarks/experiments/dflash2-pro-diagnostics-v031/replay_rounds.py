#!/usr/bin/env python3
"""Independent probability audit of actual stored q and transformed target p."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.capture.glob('round-*.pt')):
        x = torch.load(path, weights_only=False, map_location='cpu')
        p = torch.softmax(x['processed_target_logits'].double(), -1)
        assert torch.isfinite(p).all() and torch.allclose(p.sum(-1), torch.ones(p.shape[0], dtype=torch.float64))
        drafted = x['draft_sampled'][1:]
        local = x['expanded_local_pos'][:-1].long()
        positions = x['pos'][:-1]
        assert torch.equal(local, torch.arange(len(local))), local
        q_logits = x['draft_logits']
        if q_logits is None:
            assert x['draft_sample_method'] == 'greedy'
            q = torch.zeros_like(p[:-1]).scatter_(-1, drafted[:, None], 1)
            # The realized selector values are scores, not a stochastic q in
            # this mode. Softmaxing them would audit a distribution never drawn.
            tv = None
            coverage = p[:-1].gather(-1, drafted[:, None]).squeeze(-1)
            scores_exact = None
            proposal_scope = 'Dirac proposal at greedy selector token; ordinary rejection q(y)=1'
        else:
            assert x['draft_sample_method'] == 'probabilistic'
            temperature = float(x['temperature'][0])
            assert temperature > 0
            ids = x['candidate_ids'][0][local]
            scores = x['realized_scores'][0][local].double()
            actual_logits = q_logits[0][local].double()
            assert torch.isfinite(actual_logits).sum(-1).eq(16).all()
            stored_candidates = actual_logits.gather(-1, ids)
            scores_exact = bool(torch.equal(stored_candidates, scores))
            assert scores_exact, 'Rejection cache differs from the realized selector scores'
            q = torch.softmax(actual_logits/temperature, -1)
            reconstructed = torch.zeros_like(q).scatter_(-1, ids, torch.softmax(scores/temperature, -1))
            tv = .5*(q-reconstructed).abs().sum(-1)
            assert float(tv.max()) <= 1e-6
            coverage = p[:-1].gather(-1, ids).sum(-1)
            proposal_scope = 'Top16 realized selector scores, exactly one temperature division; no target top-p/top-k filter applied to q'
        overlap = torch.minimum(p[:-1], q).sum(-1)
        py = p[:-1].gather(-1, drafted[:, None]).squeeze(-1)
        qy = q.gather(-1, drafted[:, None]).squeeze(-1)
        assert (qy > 0).all()
        ratio_acceptance = torch.minimum(torch.ones_like(py), py/qy)
        positive = (p[:-1]-q).clamp_min(0)
        masses = positive.sum(-1)
        assert torch.allclose(masses, 1-overlap, atol=1e-12, rtol=0)
        rows.append(dict(round=x['round'], file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            proposal_scope=proposal_scope, request_state_ids=x['idx_mapping_np'].tolist(),
            positions=positions.tolist(), temperature=x['temperature'].tolist(), seed=x['seeds'].tolist(),
            drafted_tokens=drafted.tolist(), sampled=x['sampled_token_ids'].tolist(),
            num_sampled=x['num_sampled'].tolist(), coverage=coverage.tolist(), overlap=overlap.tolist(),
            probability_ratio_acceptance=ratio_acceptance.tolist(), residual_mass=masses.tolist(),
            q_tv=tv.tolist() if tv is not None else None, scores_exact=scores_exact))
    assert rows
    result = dict(status='PASS_CAPTURED_PROPOSAL_DISTRIBUTION_AUDIT', rows=rows,
        scope='Independent CPU FP64 audit of actual proposal law, target coverage/overlap and residual normalization; RNG decisions and paged attention remain separate gates')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], len(rows), flush=True)


if __name__ == '__main__':
    main()
