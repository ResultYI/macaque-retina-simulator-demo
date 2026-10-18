# Video diagnostics retained from the notebook validation

These are the actual diagnostic entry points used on 2026-10-18, copied out
of ignored `reports/notebook_compat/`. No optical data or checkpoints are bundled.
They intentionally refer to the original dated report directories: restore those
local assets first, or review the explicit source/output paths before adapting a
new experiment. Do not substitute a differently bound model or checkpoint.
The grouped validation's active-run pointer now lives directly under `reports/`
so it does not require the old ignored `notebook_compat` source folder.

Run scripts directly, for example:

```powershell
& $python tools/video_diagnostics/profile_actual_video_20ms.py --output reports/new_profile_20ms
& $python tools/video_diagnostics/benchmark_no_delay_20ms.py --output reports/new_no_delay_pair
& $python tools/video_diagnostics/verify_completed_video.py --run reports/your_completed_run --report reports/verification.json
```

`$python` must identify the separate validated environment. Set
`OPENBLAS_NUM_THREADS=1`, clear `PYTHONPATH`/`PYTHONHOME`, and set
`PYTHONNOUSERSITE=1` before starting Python. See
[`docs/desktop_handoff_20261018_zh.md`](../../docs/desktop_handoff_20261018_zh.md)
for versions, prerequisites, exact validation scope and source identities.

- `exclusive_advance_profile.py`: process-local nested exclusive timing, original
  function AST round-trip proof, and `finally` restoration of installed hooks.
- `profile_actual_video_20ms.py`: original versus instrumented 20–40 ms pair;
  original 0–20 ms warm-up, existing optical input and frozen background.
- `benchmark_no_delay_20ms.py`: original versus explicit no-delay guard, both
  starting from the retained strictly validated 20 ms checkpoint; no new 0.5 s run.
- `verify_completed_video.py`: read-only completed-run/checkpoint/record validation.
- `compare_completed_video_runs.py`: historical full-run comparison. Its `FOLDER`
  and receipt paths are specific to the retained experiment; it is not a generic
  new-run CLI. Only the two original process diagnostic timers are classified
  separately; other scientific/event metadata are compared exactly.
- `validate_grouped_optics.py`: original grouped-optics numerical comparison;
  requires existing MATLAB/ISET assets. Do not run merely to check installation.
- `run_clean_video_first_fault.py` and the two witness modules: optional first
  failure capture around the original driver. They never replace a failed gate.
  This entry point also requires `RETINA_FIRST_ERROR_DIR` to name an existing
  private local output directory. Failure evidence stays under ignored reports.

No script should be run automatically during clone, import of the project, or
desktop handoff. New state/Quantity buffers and pure-evaluate caches remain
unimplemented and paused at the user's request.
