# B70 C2 held-out quality fixture v2

This fixture has 48 new items across twelve distinct source contexts: eight
multi-file QueueKit agent edits, twelve standalone executable code tasks,
twelve code reviews, eight exact source retrievals and eight required API tool
calls. Two contexts are near 199K tokens. Each context is reused for four
consecutive items to exercise cold and warm prefix histories. The eight agent
edits produce adaptive, multi-step tool histories and run hidden acceptance
tests in a fresh QueueKit checkout. Source files in context are data, not
instructions.

The contexts come from the tracked `meaningful-corpus.json` at new rotations
and prompt namespaces. Their manifest is tracked in
`config/heldout_context_manifest.json`; the full frozen input archive is kept
locally at `benchmark-results/production-release-v1/fixtures/heldout-contexts-v1.tar.zst`
(SHA-256 `b98440d9762a02c19709d7c234409713969ce1bf7578d39af9d7508630e9861f`).
The task and acceptance files are pinned by `manifest.json` and
`tasks.sha256`. Neither model arm receives hidden test files or expected
answers in its prompt or tool workspace.

The candidate and the frozen offline W4A8/Q128 control must use this exact
fixture, task order, sampling, and runner. Run each arm on an exclusive
endpoint with separate AOT caches. Preserve raw rows and final project states
locally. Report every task win and loss, category counts, critical failures,
and any harness errors separately from model failures. Eight linked agent
histories plus four requests per source context cover more than a simple
single-turn code screen, but the 48 items remain an operational release check,
not a statistical guarantee of general quality.

Version 2 specifies snippet-relative review line numbers, stripped tool markers, and the rename error class. The runner caps code output below the model context limit for near-199K prompts. Version 1 remains frozen for audit and was not a completed qualification.
