# Upstream reference snapshots

`verl-v050-*` and `verl-config-v0.5.0/` are reference snapshots from the official
[verl v0.5.0 source](https://github.com/volcengine/verl/tree/v0.5.0) and published
0.5.0 wheel. `verl-wheel-source.json` records the wheel URL and SHA-256. Apache-2.0
copyright headers are retained; the full license is `LICENSE.verl`. The training
entry uses the installed locked package, not these snapshots. Snapshots support
API/config review and offline Hydra composition.

`metal-install.sh` is an unexecuted inspection copy of the official vLLM Metal
installation script, used only to identify matching release-wheel URLs. It is not
an installer invoked by this experiment. Runtime wheels and mlx-lm commit are
fixed in requirements-metal.lock. See the official
[vLLM Metal repository](https://github.com/vllm-project/vllm-metal).

Data/model licenses are independent of framework code licenses. PsyDT's own
pinned dataset card and source manifest are preserved under artifacts/sources;
see docs/DATA.md. No patient records or fabricated learned adapters are shipped.
