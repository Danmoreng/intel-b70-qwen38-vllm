# Previous vLLM0.31 / oneDNN3.13.4 production release

This snapshot retains the previously qualified immutable EXL3 image
`sha256:36e41ea027758060f9352bdd7f9ed47de5143d0424b238653960dbf8357c176a`,
its policy, native source snapshot and unchanged Python runtime module bytes.
It includes the approved12GiB RAM cache and explicit LBNHC layout.

The RAM manifest and its image attestation are relocated to this directory.
The manifest selects the archived `runtime/` here, so later changes to the
current runtime do not replace the old0.31 adapter during rollback. Policy,
image, checkpoint, middleware and module integrity checks remain enforced.
The original public source state is commit`ebf24eda3db60371735d0f09a20dbdb566ac2be8`.

`scripts/run-server-exl3-v4-rollback.sh --check-only` performs the strict
preflight. The same script without that argument starts the archived profile.
Stop the current text service before starting a rollback worker; both profiles
use the same GPU and API port by default. The default power cap remains180W.
