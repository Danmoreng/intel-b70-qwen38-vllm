#!/usr/bin/env python3
"""Summarize non-overlapping trace boundaries without summing nested scopes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    output = {'scope': 'profiled diagnostic; never scored serving throughput',
              'timing_note': 'XPU event intervals include submission gaps. Child intervals overlap their parents and must not be added.',
              'arms': []}
    for path in sorted(args.root.glob('arm-*/step-trace-*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        metadata = defaultdict(list)
        for row in rows:
            if row['kind'] == 'attention_metadata':
                metadata[row['step']].append(row)
        for count in (1, 4):
            valid = {step for step, meta in metadata.items() if len(meta) == 1
                     and meta[0]['actual_requests'] == count
                     and meta[0]['scheduled_query_lengths'] == [5] * count
                     and meta[0]['actual_query_rows'] == meta[0]['padded_query_rows'] == count * 5}
            selected = [row for row in rows if row['step'] in valid]
            lengths = [length for step in valid for length in metadata[step][0]['seq_lens'][:count]]
            summaries = []
            for kind, component in (('execute_model', 'target_and_prepare'),
                    ('graph', 'target_and_prepare'), ('sample_tokens', 'sample_commit_draft'),
                    ('propose', 'draft'), ('__call__', 'rejection_sampler')):
                group = [row for row in selected if row['kind'] == kind and row.get('component') == component]
                if group:
                    summaries.append({'kind': kind, 'component': component, 'count': len(group),
                        'median_stream_ms': statistics.median(row['stream_elapsed_ms'] for row in group),
                        'median_host_ms': statistics.median((row['host_end_ns'] - row['host_start_ns']) / 1e6 for row in group)})
            graph_modes = Counter()
            for row in selected:
                if row['kind'] == 'graph':
                    graph_modes[(row['component'], row['mode'], row['replay'],
                                 json.dumps(row.get('descriptor'), sort_keys=True), row['replay_id'])] += 1
            # Subtract children only within the SAME measured step and only if
            # their host intervals prove ordered nesting. Never subtract pooled
            # medians or add a parent to its children.
            residuals = defaultdict(list)
            for step in valid:
                step_rows = [row for row in selected if row['step'] == step]
                for parent_kind, child_specs, label in (
                    ('execute_model', [('graph', 'target_and_prepare')], 'prepare_and_submission_outside_target_graph'),
                    ('sample_tokens', [('propose', 'draft'), ('__call__', 'rejection_sampler')], 'target_head_commit_and_submission'),
                ):
                    parents = [row for row in step_rows if row['kind'] == parent_kind]
                    children = [row for row in step_rows
                                if (row['kind'], row.get('component')) in child_specs]
                    if len(parents) != 1 or len(children) != len(child_specs):
                        continue
                    parent = parents[0]
                    children.sort(key=lambda row: row['host_start_ns'])
                    nested = all(parent['host_start_ns'] <= row['host_start_ns'] <= row['host_end_ns'] <= parent['host_end_ns']
                                 for row in children)
                    ordered = all(a['host_end_ns'] <= b['host_start_ns'] for a, b in zip(children, children[1:]))
                    elapsed = parent['stream_elapsed_ms'] - sum(row['stream_elapsed_ms'] for row in children)
                    if nested and ordered and elapsed >= 0:
                        residuals[label].append(elapsed)
            output['arms'].append({'arm': path.parent.name, 'concurrency': count,
                'actual_and_padded_query_rows': count * 5, 'valid_steps': len(valid),
                'min_true_kv': min(lengths) if lengths else None,
                'max_true_kv': max(lengths) if lengths else None,
                'boundaries': summaries,
                'same_step_stream_residuals': {label: {'count': len(values), 'median_ms': statistics.median(values)}
                                               for label, values in residuals.items()},
                'residual_note': 'Ordered nested scopes on the instrumented current stream; residual includes device work and host submission gaps, not pure CPU time.',
                'graphs': [dict(component=key[0], mode=key[1],
                    replay=key[2], descriptor=json.loads(key[3]), replay_id=key[4], count=value)
                    for key, value in graph_modes.items()]})
    (args.root / 'trace-analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
