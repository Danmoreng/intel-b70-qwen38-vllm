"""Same-window speculative counters; round equivalents are not GPU timings."""

from typing import Any


def summarize_overlap(samples: list[dict[str, Any]], start: float, stop: float,
                      concurrency: int, speculative_tokens: int = 4) -> dict[str, Any]:
    window = [row for row in samples if start <= row['elapsed_s'] <= stop]
    valid = [row for row in window if 'error' not in row]
    result = {
        'fully_overlapped_sample_count': len(valid),
        'fully_overlapped_speculative_tokens_per_request': speculative_tokens,
        'fully_overlapped_draft_tokens': None,
        'fully_overlapped_accepted_tokens': None,
        'fully_overlapped_speculative_acceptance': None,
        'fully_overlapped_round_equivalents': None,
        'fully_overlapped_round_equivalent_ms': None,
        'fully_overlapped_token_accounting_residual': None,
        'fully_overlapped_counter_window_valid': False,
        'fully_overlapped_counter_window_issues': [],
    }
    issues = result['fully_overlapped_counter_window_issues']
    if len(valid) < 2:
        issues.append('insufficient_samples')
        return result
    first, last = valid[0], valid[-1]
    seconds = last['elapsed_s'] - first['elapsed_s']
    drafted = last['draft_tokens'] - first['draft_tokens']
    accepted = last['accepted_tokens'] - first['accepted_tokens']
    generated = last['generation_tokens'] - first['generation_tokens']
    residual = generated - accepted - drafted / speculative_tokens
    rounds = drafted / (concurrency * speculative_tokens)
    if len(valid) != len(window):
        issues.append('metric_sample_error')
    if any(row['running'] != concurrency or row['waiting'] != 0 for row in valid):
        issues.append('nonconstant_occupancy')
    if any(row['prefill_computed'] != first['prefill_computed'] for row in valid):
        issues.append('prefill_inside_window')
    if any(b[key] < a[key] for a, b in zip(valid, valid[1:])
           for key in ('draft_tokens', 'accepted_tokens', 'generation_tokens')):
        issues.append('counter_reset')
    if seconds <= 0 or drafted <= 0 or not 0 <= accepted <= drafted:
        issues.append('invalid_counter_delta')
    if drafted % (concurrency * speculative_tokens):
        issues.append('partial_round_equivalent')
    if residual != 0:
        issues.append('token_accounting_mismatch')
    result.update({
        'fully_overlapped_draft_tokens': drafted,
        'fully_overlapped_accepted_tokens': accepted,
        'fully_overlapped_speculative_acceptance': accepted / drafted if drafted > 0 else None,
        'fully_overlapped_round_equivalents': rounds,
        'fully_overlapped_round_equivalent_ms': 1000 * seconds / rounds
        if rounds > 0 and seconds > 0 and not issues else None,
        'fully_overlapped_token_accounting_residual': residual,
        'fully_overlapped_counter_window_valid': not issues,
        'fully_overlapped_sample_start_s': first['elapsed_s'],
        'fully_overlapped_sample_end_s': last['elapsed_s'],
    })
    return result


def pool_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get('fully_overlapped_counter_window_valid')]
    seconds = sum(row['fully_overlapped_decode_sampled_s'] for row in valid)
    drafted = sum(row['fully_overlapped_draft_tokens'] for row in valid)
    accepted = sum(row['fully_overlapped_accepted_tokens'] for row in valid)
    rounds = sum(row['fully_overlapped_round_equivalents'] for row in valid)
    return {
        'fully_overlapped_valid_counter_windows': len(valid),
        'fully_overlapped_invalid_counter_windows': len(rows) - len(valid),
        'fully_overlapped_draft_tokens': drafted,
        'fully_overlapped_accepted_tokens': accepted,
        'fully_overlapped_speculative_acceptance': accepted / drafted if drafted else None,
        'fully_overlapped_round_equivalents': rounds,
        'fully_overlapped_round_equivalent_ms': 1000 * seconds / rounds if rounds else None,
        'fully_overlapped_token_accounting_residual': sum(
            row['fully_overlapped_token_accounting_residual'] for row in valid),
    }
