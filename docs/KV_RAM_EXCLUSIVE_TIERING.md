# Exclusive GPU/RAM tiering: design boundary

This is a design note, not implemented behavior or a release claim. It explains
the remaining distinction between the passing 292,800-token CPU-retention test
and a guarantee for alternating maximum-size independent agent histories.

## Why a capacity sum is insufficient

The pinned native OffloadingConnector eagerly copies reusable GPU pages to RAM.
The two tiers are inclusive: loading a prefix does not retire its CPU copies,
and evicting a completed GPU prefix does not automatically move all its victim
pages into newly freed CPU slots. ARC can preserve a frequently reused older
history, but replacement-policy parameters cannot turn this into exclusive
tiering. Two full CPU replicas alone exceed the requested 12 GiB budget.

A second history can remain on GPU while the first is in RAM, yet returning to
the first can overwrite GPU pages of the second without a complete CPU replica
of that second history. A successful one-way handoff therefore does not prove
the requested two-history behavior. The `two-full` test checks both returns
and cancels immediately on an insufficient prefix hit.

## Proposed bounded movement

Implement a joint cache planner that observes GPU victims **before** their
block identity is reassigned or the model runner zeros their contents. Operate
on the native hybrid groups and page mappings; attention prefixes and their
matching recurrent checkpoints must move together logically. Native hashes,
group IDs and request/cache isolation must remain unchanged.

For a large handoff with sufficient combined capacity, use a bounded page
pipeline rather than copying a whole additional history into temporary RAM:

1. Lease only inactive GPU victim pages; active-request references are excluded.
2. Store a bounded set of those victims into currently free CPU slots and wait
   for transfer completion.
3. Load incoming CPU pages into the released GPU slots. Retire their CPU copies
   only after the GPU copy is valid, making those CPU slots available for the
   next group of victims.
4. Repeat until the incoming history is GPU-resident and the displaced history
   is retained in the other tier. Publish the completed load to the scheduler.

The existing 128 MiB staging bound remains in force. There must be no full-
history scratch allocation. A joint lookup must also understand a history
temporarily distributed across both tiers; a pure CPU-prefix lookup cannot
represent that state. The admission reservation includes all in-flight pages.

This needs coordinated changes to block ownership, connector metadata and
worker ordering. Merely wrapping a GPU-eviction callback is insufficient:
vLLM's asynchronous zeroing may already be in flight from an earlier step.
Generation checks and explicit stream fences must ensure that victim data is
copied before reuse and incoming data is not zeroed after loading.

Cancellation cannot abandon a half-swap and silently lose both cache copies.
Either complete the bounded movement or retain enough ownership information
to expose every still-valid page through the joint lookup. No page can be
relabelled as ready while its transfer is unfinished.

## Acceptance before production

- With the same 12 GiB physical host-cache budget, run A, B, A, B for two distinct
  262K histories, then several further alternating rounds. Every return must
  reuse the nearly complete prefix and retrieve all history-only check records.
- Verify actual host availability, resident cache and staging peaks, including
  mixed small requests; do not infer these from virtual mmap size alone.
- Verify cancellation at every transfer stage, zeroing/transfer races, shared
  prefix ownership and recurrent-state correctness. Preserve request completion
  and no-preemption guarantees from the admission policy.
- Pin and attest the resulting runtime separately, compare with the native
  control and retain the existing production service as rollback.

Increasing the inclusive CPU pool enough for two full replicas is a simpler
alternative, but exceeds the user's RAM budget and leaves too little headroom
on this machine. An SSD-backed additional tier would be a different design
with separate I/O and storage limits; the current candidate does not use it.
