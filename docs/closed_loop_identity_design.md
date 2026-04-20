# Closed-Loop Identity-Preserving Kilosort4

## Why this exists
Closed-loop visual neuroscience experiments often need to treat multiple recording blocks as one iterative workflow instead of independent spike-sorting jobs. The main goal of this extension is to preserve unit correspondence across loops well enough that later blocks can reuse earlier models, templates, and stimulus logic.

## Core idea
The default Kilosort4 path is left unchanged unless closed-loop mode is explicitly enabled.

In closed-loop mode:
- loop 1 still runs standard KS4
- loop 1 exports a `prior_bundle.npz` with canonical templates and alignment metadata
- loop N aligns the saved prior bundle into the current recording using a rigid depth shift
- carried-over units are matched directly from the aligned prior templates
- carried-over identities do not go through the final reclustering / merging step
- standard KS4 discovery still runs for the current recording, and genuinely new units are appended with new global ids

This produces two identity layers:
- local cluster ids remain contiguous for Phy/export compatibility
- global unit ids are saved separately and are stable across loops

## Saved artifacts
Each run now writes:
- `prior_bundle.npz`: carry-forward template state for the next loop
- `cross_loop_unit_map.csv`: local cluster id to global unit id plus carry/new/stale status
- `cross_loop_summary.json`: loop-level summary including the rigid shift estimate
- `cross_loop_debug.npz`: visualization-ready arrays for cross-session inspection
- `cluster_global_ids.npy` and `spike_global_ids.npy`: explicit global-id exports alongside the standard KS4 files

## Two-session walkthrough
Typical usage:
1. Run loop 1 with standard settings. The run still exports `prior_bundle.npz`.
2. Run loop 2 with:
   - `closed_loop_identity_mode="preserve"`
   - `closed_loop_prior_path=<loop1 results>/prior_bundle.npz`
3. Inspect:
   - `cross_loop_unit_map.csv` for carried/new/stale units
   - `cross_loop_summary.json` for the rigid alignment shift
   - `cross_loop_debug.npz` for plotting templates and alignment diagnostics
4. Open `docs/tutorials/closed_loop_two_session_walkthrough.ipynb` and point it at the two results folders.

## Important implementation choices
- V1 supports rigid cross-session depth alignment only.
- Carried-over units are never sent through the final KS4 reclustering path in preserve mode.
- New units are still discovered with the normal KS4 discovery branch and appended afterward.
- Template adaptation is conservative: carried templates are only updated when enough current-session evidence exists.
- Stale units are preserved in the next `prior_bundle.npz` even if they are not active in the current export.

## Current limits
- Cross-session alignment is rigid, not nonrigid.
- Carried-unit adaptation currently uses current-session duplicate discovery templates as the update target.
- The notebook is intended as a lightweight analysis walkthrough, not a complete curation GUI.

## Future extensions
- nonrigid cross-session alignment
- stronger spike-level residual discovery branch
- richer confidence scoring for carry-over matches
- explicit loop/session metadata in the public API
