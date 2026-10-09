"""Independent exact finite-vocabulary reference, no vLLM imports."""
import math


def exact_first_token(p, q):
    assert len(p) == len(q)
    assert all(x >= 0 for x in p+q)
    assert abs(sum(p)-1) < 1e-12 and abs(sum(q)-1) < 1e-12
    accepted = [min(a, b) for a, b in zip(p, q)]
    residual = [max(a-b, 0) for a, b in zip(p, q)]
    reject_mass = 1-sum(accepted)
    if reject_mass > 1e-14:
        residual_sum = sum(residual)
        assert abs(residual_sum-reject_mass) < 1e-12
        result = [a+reject_mass*r/residual_sum for a, r in zip(accepted, residual)]
    else:
        result = accepted
    assert max(abs(a-b) for a, b in zip(result, p)) <= 1e-12
    return dict(output=result, acceptance=sum(accepted),
                acceptance_by_proposal=[min(1, a/b) if b else None for a, b in zip(p, q)],
                residual=[r/sum(residual) for r in residual] if sum(residual) else None)


CASES = {
    'equal': ([.1, .2, .3, .4], [.1, .2, .3, .4]),
    'disjoint': ([.4, .6, 0., 0.], [0., 0., .3, .7]),
    'near_equal': ([.1001, .1999, .3, .4], [.1, .2, .3, .4]),
    'overlap': ([.05, .65, .2, .1], [.55, .1, .3, .05]),
}


def gpu_greedy_test():
    """Independent exact T0 prefix oracle, C4 slots and changed graph inputs."""
    import torch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    device='xpu';n=4;steps=7
    logits=torch.zeros(n*(steps+1),4,device=device,dtype=torch.float32)
    proposed=torch.zeros(n*(steps+1),device=device,dtype=torch.int64)
    slots=torch.tensor([2,0,3,1],device=device,dtype=torch.int32)
    expanded=slots.repeat_interleave(steps+1)
    local=torch.arange(steps+1,device=device,dtype=torch.int32).repeat(n)
    cu=torch.arange(n+1,device=device,dtype=torch.int32)*(steps+1)
    positions=torch.arange(n*(steps+1),device=device,dtype=torch.int64)+1000
    temperatures=torch.zeros(n,device=device,dtype=torch.float32)
    seeds=torch.arange(n,device=device,dtype=torch.int64)+20261010
    prefixes=[7,0,3,6]
    def update(offset):
        target=[[(r+j+offset)%4 for j in range(steps+1)] for r in range(n)]
        values=torch.zeros(n,steps+1,4)
        draft=torch.zeros(n,steps+1,dtype=torch.int64)
        for r in range(n):
            for j in range(steps+1):values[r,j,target[r][j]]=10
            for j in range(steps):draft[r,j+1]=target[r][j] if j<prefixes[r] else (target[r][j]+1)%4
        logits.copy_(values.reshape(-1,4));proposed.copy_(draft.flatten())
        return target,draft
    def run():
        return rejection_sample(logits,None,proposed,cu,positions,slots,expanded,local,
            temperatures,seeds,steps,None,use_fp64=False)
    def check(result,target,draft):
        sampled,count=[v.cpu() for v in result]
        assert count.tolist()==[k+1 for k in prefixes]
        for r,k in enumerate(prefixes):
            expected=draft[r,1:k+1].tolist()+[target[r][k]]
            assert sampled[r,:k+1].tolist()==expected,(r,sampled[r,:k+1],expected)
    target,draft=update(0);check(run(),target,draft)
    torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):result=run()
    for offset in [1,2]:
        target,draft=update(offset)
        slots.copy_(slots.roll(1));expanded.copy_(slots.repeat_interleave(steps+1))
        graph.replay();check(result,target,draft)
    return dict(status='PASS_EXACT_T0_REJECTION_AND_GRAPH_SLOT_REPLAY',
        requests=4,steps=7,prefixes=prefixes,graph_changed_input_replays=2,
        scope='Exact argmax and first/middle/late rejection/all-accepted bonus. EOS and frontend token limits are separately covered by serving smokes.')


def gpu_test():
    import torch
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    # Freeze familywise significance before any GPU outputs are inspected.
    count = 32768
    alpha = .001
    comparisons = len(CASES)*2*9  # four q and four output bins plus acceptance
    bound = math.sqrt(math.log(2*comparisons/alpha)/(2*count))
    results = []
    device = 'xpu'
    for name, (p, q) in CASES.items():
        reference = exact_first_token(p, q)
        for temp in (1., 1.95):
            idx = torch.arange(count, device=device, dtype=torch.int32)
            # Permuted persistent request slots differ from batch order.
            slots = idx.flip(0).contiguous()
            seeds = torch.arange(count, device=device, dtype=torch.int64)*97+554433
            temperatures = torch.full((count,), temp, device=device)
            positions = torch.full((count,), 33001, device=device, dtype=torch.int64)
            ids = torch.arange(4, device=device).expand(count, 1, 4).contiguous()
            q_scores = torch.tensor([math.log(v)*temp if v else -math.inf for v in q], device=device)
            scores = q_scores.expand(count, 1, 4, 4).contiguous()
            proposal = torch.empty((count, 1), device=device, dtype=torch.int64)
            cache = torch.full((count, 1, 4), -math.inf, device=device)
            sampler = CandidateSampler(count, 1, 4, torch.device(device))
            sampler.sample(ids, scores, count, positions, slots,
                           temperatures, seeds, proposal, cache, False)
            target = torch.tensor([math.log(v) if v else -math.inf for v in p], device=device)
            target = target.expand(count*2, 4).contiguous()
            drawn = torch.zeros(count*2, device=device, dtype=torch.int64)
            # vLLM ABI: row zero is the anchor; proposal j lives at row j+1.
            drawn[1::2] = proposal[:, 0]
            cu = torch.arange(0, count*2+1, 2, device=device, dtype=torch.int32)
            expanded_slots = slots.repeat_interleave(2)
            local_pos = torch.tensor([0, 1], device=device, dtype=torch.int32).repeat(count)
            pos = torch.tensor([33000, 33001], device=device, dtype=torch.int64).repeat(count)
            sampled, lengths = rejection_sample(target, cache, drawn, cu, pos,
                slots, expanded_slots, local_pos, temperatures, seeds, 1,
                synthetic_conditional_rates=None, use_fp64=False, use_block_verification=False)
            torch.xpu.synchronize()
            proposal_frequency = torch.bincount(proposal[:, 0], minlength=4).double().cpu()/count
            output_frequency = torch.bincount(sampled[:, 0], minlength=4).double().cpu()/count
            acceptance = float((lengths == 2).double().mean().cpu())
            proposal_error = max(abs(a-b) for a, b in zip(proposal_frequency.tolist(), q))
            output_error = max(abs(a-b) for a, b in zip(output_frequency.tolist(), p))
            acceptance_error = abs(acceptance-reference['acceptance'])
            cache_tv = float((torch.softmax(cache[slots, 0].double()/temp, -1)-torch.tensor(q, device=device)).abs().sum(-1).max().cpu()/2)
            assert max(proposal_error, output_error, acceptance_error) <= bound, (name, temp, proposal_error, output_error, acceptance_error, bound)
            assert cache_tv <= 1e-6, (name, cache_tv)
            if name == 'equal':
                assert bool((lengths == 2).all()), 'p=q must fully accept'
            if name == 'disjoint':
                assert bool((lengths == 1).all()), 'disjoint supports must reject first token'
            results.append(dict(case=name, temperature=temp, count=count,
                expected=reference, proposal_frequency=proposal_frequency.tolist(),
                first_output_frequency=output_frequency.tolist(), acceptance=acceptance,
                max_bin_error=max(proposal_error, output_error), acceptance_error=acceptance_error,
                q_cache_tv=cache_tv))
    return dict(status='PASS_GPU_PROPOSAL_AND_ORDINARY_REJECTION_STATISTICS',
        familywise_alpha=alpha, simultaneous_hoeffding_bound=bound,
        cases=results, scope='Exact four-token distributions, one speculative step, permuted persistent slots; not EOS/limit, multistep or graph-replay qualification')


def graph_test():
    """Actual Top16 conditional chain, changing slots/candidates, seven steps."""
    import torch
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    device = 'xpu'
    c, steps, vocab = 4, 7, 32
    sampler = CandidateSampler(c, steps, 16, torch.device(device))
    ids = torch.arange(16, device=device).expand(c, steps, 16).contiguous()
    scores = torch.sin(torch.arange(c*steps*16*16, device=device).float()*.037).reshape(c, steps, 16, 16).contiguous()
    slots = torch.tensor([3, 0, 2, 1], device=device, dtype=torch.int32)
    expanded = slots.repeat_interleave(steps)
    position = torch.arange(33001, 33001+steps, device=device).repeat(c)
    temp = torch.full((c,), 1., device=device)
    seed = torch.arange(c, device=device, dtype=torch.int64)+11177
    proposed = torch.empty((c, steps), device=device, dtype=torch.int64)
    cache = torch.full((c, steps, vocab), -math.inf, device=device)
    cu = torch.arange(0, c*(steps+1)+1, steps+1, device=device, dtype=torch.int32)
    local = torch.arange(steps+1, device=device, dtype=torch.int32).repeat(c)
    pos = torch.arange(33000, 33000+steps+1, device=device).repeat(c)
    mapped = slots.repeat_interleave(steps+1)

    def run():
        sampler.sample(ids, scores, c, position, expanded, temp, seed, proposed, cache, False)
        target = torch.empty((c, steps+1, vocab), device=device)
        target[:, :steps] = cache[slots]
        target[:, steps] = 0
        drawn = torch.empty((c, steps+1), device=device, dtype=torch.int64)
        drawn[:, 0] = 17
        drawn[:, 1:] = proposed
        return rejection_sample(target.flatten(0, 1), cache, drawn.flatten(), cu,
            pos, slots, mapped, local, temp, seed, steps,
            synthetic_conditional_rates=None, use_fp64=False, use_block_verification=False)

    for _ in range(3):
        run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        sampled, lengths = run()
    cases = []
    for iteration, offset in enumerate((0, 16, 0)):
        ids.copy_((torch.arange(16, device=device)+offset).expand_as(ids))
        slots.copy_(torch.tensor([[3,0,2,1],[1,3,0,2],[2,1,3,0]][iteration], device=device))
        expanded.copy_(slots.repeat_interleave(steps)); mapped.copy_(slots.repeat_interleave(steps+1))
        seed.add_(1000)
        graph.replay(); torch.xpu.synchronize()
        assert bool((lengths == steps+1).all()), lengths
        assert torch.equal(sampled[:, :steps], proposed)
        realized = sampler.scores.cpu().double()
        proposal_cpu = proposed.cpu()
        ids_cpu = ids.cpu()
        scores_cpu = scores.cpu().double()
        cached = cache[slots].cpu().double()
        tv_max = 0.
        for req in range(c):
            previous = 0
            for step in range(steps):
                expected = scores_cpu[req, step, previous]
                assert torch.equal(expected, realized[req, step])
                assert torch.equal(cached[req, step, ids_cpu[req,step]], expected)
                assert torch.isfinite(cached[req,step]).sum() == 16
                dense = torch.zeros(vocab, dtype=torch.float64)
                dense[ids_cpu[req,step]] = torch.softmax(expected, -1)
                tv = float((torch.softmax(cached[req,step], -1)-dense).abs().sum()/2)
                tv_max = max(tv_max, tv)
                previous = int((ids_cpu[req,step] == proposal_cpu[req,step]).nonzero()[0])
        assert tv_max <= 1e-6
        cases.append(dict(iteration=iteration, candidate_offset=offset, q_tv_max=tv_max,
                          slots=slots.cpu().tolist(), useful_lengths=lengths.cpu().tolist()))
    return dict(status='PASS_SEVEN_STEP_TOP16_PROPOSAL_Q_CACHE_REJECTION_GRAPH', cases=cases,
        scope='Actual selector walk/cache/rejection kernels with independent conditional score indexing, candidate eviction and permuted persistent slots; no model attention or serving token-limit qualification')


if __name__ == '__main__':
    import json
    print(json.dumps({name: exact_first_token(p, q) for name, (p, q) in CASES.items()}, indent=2))
