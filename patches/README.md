# Opt-In Upstream Patch Overlays

`layerstream-plugin` builds against **100% stock upstream** `llama.cpp` and `mesh-llm` by default.
This directory provides an opt-in, component-level patch overlay for clean A/B benchmarking
(`stock` vs. `patched`) without maintaining permanent forks.

## Selector Syntax (`LAYERSTREAM_PATCHES`)

Set `LAYERSTREAM_PATCHES` at build time using a comma-separated list of component tokens:

| Value | Effect |
|---|---|
| `none` *(default / unset)* | Stock upstream — no patches applied. |
| `all` | Apply all component patch sets (`llama`, `mesh`). |
| `llama` | Apply only `patches/llama.cpp/*.patch`. |
| `mesh` | Apply only `patches/mesh-llm/*.patch` (via Cargo `[patch]` override). |
| `all,-llama` | Apply every component **except** `llama.cpp`. |
| `llama,mesh` | Explicit include list. |

Every compiled binary stamps its active patch set into `layerstream_core::ACTIVE_PATCHES`
(and startup logs when non-empty) so benchmark outputs record `stock` vs. `patched`.

## Patch Rules

1. **Upstream first:** every `.patch` file must reference its upstream tracking issue or PR
   (`upstream: Owner/repo#123`) in its header comment.
2. **Keep stacks small:** target 0–3 patches per component and delete a patch as soon as
   upstream merges it.
3. **CI drift guard:** `CI/check` runs `git apply --check` on `patches/llama.cpp/*.patch`
   against the pinned `third_party/llama.cpp` on every PR.
