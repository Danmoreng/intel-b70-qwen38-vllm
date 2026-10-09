#!/usr/bin/env python3
"""Independent probability audit of actual stored q and transformed target p."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from rng_reference import acceptance_uniform,known_answers,sampled_winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    assert known_answers()==3
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
        # Replay the actual ordinary-rejection prefix with independent scalar
        # Philox, including the installed FP32 uniform conversion. Require
        # margin from the threshold before claiming a discrete exact match.
        assert float(x['temperature'][0])>0, 'This actual-round oracle expects stochastic target sampling'
        uniforms=[acceptance_uniform(int(x['seeds'][0]),int(pos)) for pos in positions]
        log_margins=[float(a.log()-b.log())-math.log(u) for a,b,u in zip(py,qy,uniforms)]
        assert all(abs(v)>1e-5 for v in log_margins),'Threshold too close for this FP64-vs-FP32 oracle'
        prefix=0
        for margin in log_margins:
            if margin<=0:break
            prefix+=1
        assert int(x['num_sampled'][0])==prefix+1, (x['round'],prefix,x['num_sampled'])
        assert torch.equal(x['sampled_token_ids'][0,:prefix],drafted[:prefix])
        if prefix<len(drafted):assert x['sampled_token_ids'][0,prefix]!=drafted[prefix]
        gumbel_replay=None
        if 'use_fp64_gumbel' in x:
            seed=int(x['seeds'][0]);fp64=bool(x['use_fp64_gumbel'])
            proposal_checks=[]
            if q_logits is not None:
                for j in range(len(drafted)):
                    winner,margin=sampled_winner(zip(ids[j].tolist(),(scores[j]/temperature).tolist()),
                        seed,int(positions[j])+(1<<30),fp64)
                    assert margin>1e-3,'Proposal winner too close for independent math-log comparison'
                    assert winner==int(drafted[j]),(x['round'],j,winner,int(drafted[j]))
                    proposal_checks.append(dict(step=j,token=winner,winner_margin=margin))
            if prefix==len(drafted):
                logits=p[prefix].log()
            else:
                # The Dirac residual is target logits with its proposal
                # removed; normal q uses the independently normalized p-q.
                logits=(p[prefix]-q[prefix]).clamp_min(0).log()
            support=torch.where(torch.isfinite(logits))[0]
            winner,margin=sampled_winner(zip(support.tolist(),logits[support].tolist()),
                seed,int(x['pos'][prefix]),fp64)
            assert margin>1e-3,'Residual/bonus winner too close for independent math-log comparison'
            assert winner==int(x['sampled_token_ids'][0,prefix]),(x['round'],winner,x['sampled_token_ids'])
            gumbel_replay=dict(proposal_draws=proposal_checks,residual_or_bonus_token=winner,
                winner_margin=margin,use_fp64=fp64,draft_salt=1<<30,target_salt=0)
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
            acceptance_uniforms=uniforms,acceptance_log_margins=log_margins,
            independently_replayed_accepted_prefix=prefix,
            gumbel_replay=gumbel_replay,
            q_tv=tv.tolist() if tv is not None else None, scores_exact=scores_exact))
    assert rows
    result = dict(status='PASS_CAPTURED_PROPOSAL_DISTRIBUTION_AUDIT', rows=rows,
        reference_sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['replay_rounds.py','rng_reference.py']},random123_known_answer_vectors=3,
        scope='Independent CPU FP64 proposal/coverage/overlap/residual and Philox acceptance-prefix replay. Captures attesting the Gumbel mode also replay actual proposal and residual/bonus winners with an independent math-log oracle. Older captures lack that mode and do not claim this gate. Paged attention is separate.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], len(rows), flush=True)


if __name__ == '__main__':
    main()
