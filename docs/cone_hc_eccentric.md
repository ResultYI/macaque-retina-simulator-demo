# Eccentric Cone–H1 initialization and full-chain entry

The current S-priority H2 research entry is
`configurations/research/cone_hc_h2_priority.json`, dispatched by the same
`RetinaBuilder.build_research_cone_hc(root, configuration)` entry point. It
retains this H1 model and applies `C_ji * gamma_i / sum(C_jk * gamma_k)` to
complete legal fine H2 sources, with gamma L/M/S = 1/1/10 and evidence class
`published_model_borrowed`. Sparse weights are cached before runtime; original
B, H2 dynamics and all downstream parameters remain unchanged.

The completed matched L/M/S acceptance is in
`reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/h2_priority_acceptance/RESULT.json`.
Weight implementation and engineering acceptance passed; full-chain interface
acceptance is conditional on the declared finite boundary. Spectral amplitude
consistency remains uncalibrated. FIT_READY is unchanged. The final report
contains the statistics, raw outputs and comparison figure. The sections below
retain the prior uniform-H2 entry and its historical acceptance.

For the accepted S-isolation protocol, construct `model.exterior_cache()` and
use `model.isolation('S', cache).runner()`. This supplies the two actual missing
S members with time-varying Cone inputs while retaining fixed exterior HC
feedback. The cache covers only the declared 1000-to-1010 native Td step over
20–120 ms. It is not a general arbitrary-stimulus exterior network. The default
no-argument Builder and frozen historical configurations remain unchanged.

Execution status: **conditional engineering acceptance passed on 2026-11-04**.
After the user resumed execution, six minimal unit checks, spatial preparation,
five local native diagnostics and a 50 ms full-chain/checkpoint check completed.
H1's accepted homogeneous temporal calibration is installed in this explicit
entry. H2's stronger-S relation remains unresolved; FIT_READY remains false.
Current evidence is in the existing HC report's `eccentric_acceptance/RESULT.json`
and `FINAL_REPORT.md`. The configuration's implementation-status string records
its creation phase; its validated bytes are retained to preserve asset identity.

## Configuration and reference geometry

`configurations/research/HC_eccentric_biexponential.json` defines the current
task contract. Each fine H1 uses its actual OPL anchor relative to the declared
fovea. Cone distances use the original OPL pedicles:

```text
R = existing_selected_profile.contact_radius_mm(eccentricity_mm, 'H1')
lambda_short_mm = R / ln(10)
lambda_long_mm = 0.300 * R / R_at_4mm
support_radius_mm = 11.75637 * lambda_long_mm
A = 0.15 * exp(-distance/lambda_short)/sum_short
  + 0.85 * exp(-distance/lambda_long)/sum_long
```

The user's subsequent instruction supersedes the historical reference-domain
freeze. An explicitly identified evaluation view of the **same profile and
formula** permits the 4 mm reference. It changes no coefficient, central join,
cell coordinate, contact count or geometry-generation domain. Nonpositive or
nonfinite evaluation still fails. The authorization, original domain, reference
value and profile/implementation identities are recorded in the prepared ledger.

The given support factor is used exactly. Its continuum tail is recorded at
that rounded factor alongside the nominal `1e-4` budget; the tail is not a
closed-loop spatial-error bound. Central 0–6° spatial parameters remain an
engineering extrapolation rather than independent physiological estimates.

## Preparation, separate from execution

Preparation was executed once with the existing numerical environment:

```powershell
& 'D:\anaconda\envs\snn_env\python.exe' -X utf8 -m tools.prepare_hc_eccentric
```

The preparation reuses the source-bound GIN69 local graph's existing Cone/HC
members, original anatomical contacts and previously typed outer Cone envelope.
It performs no geometry regeneration, optical calculation, temporal fit or
neural simulation. Existing destination directories are rejected rather than
overwritten; failed receipts remain available for diagnosis.

For every H1, both exponentials are normalized separately over the complete
eligible L/M support. The short component has no anatomical contact mask. Fine
A is then cropped with explicit held mass, and projected with the original
all-member denominators. Group centroids never enter weight construction. B is
verified against the complete anatomical fine projection, then the unchanged
existing group B is retained. H2 A is also retained exactly. Two complete fine
H1 rows at different eccentricities are saved as formula witnesses.

Outputs live under the existing HC report's `eccentric_assets/`, with a source
receipt, fine/group CSR operators, member identities, per-H1 spatial parameters,
held/unsupported mass and the retained H2 homotypic operator. An incomplete
source envelope stops preparation without expansion or inferred labels.

## Builder and native parameter binding

```python
from retina_sim.assembly import RetinaBuilder

model = RetinaBuilder.build_research_cone_hc(
    repo_root, "configurations/research/cone_hc_eccentric.json"
)
runner = model.runner()  # existing CPU full-chain scheduler
```

This is an explicit new entry. The no-argument historical functional entry
retains its original constant-scale kernel and 20 ms time constant.
The new entry fails if its prepared assets are absent or stale; it never silently
uses uniform or the old constant-scale kernel.

The completed `CALIBRATION_METRICS.json` is read by hash. Its recorded training,
held-out and polarity gate determines the only allowed temporal change. No fit
is invoked. The current saved decision accepts `tau_h=30.865749126454666 ms`;
a rejected calibration would retain 20 ms.

`EccentricHCNative` reuses the existing H1 native coefficient equations. It
rescales only H1's final `(a_h,b_h)` pair by `20/tau_h`. The immutable source
parameter dictionary remains the source transcription, while
`effective_h1_parameters`, `tau_h_ms`, the native identity and checkpoint bind
the effective calibrated value. This avoids changing a global source constant
that H2 also borrows. Only the full joint CPU reference advancement is admitted;
the separate source-parameter multirate candidate is not an execution option.

The same A enters pooled release, pooled Vbar and spatial adaptation. Cone
release reads the current local B times the current HC state. The 2.82 ms delay
remains observation-only. No spatial searches or exponentials run in the RHS.

The retained full-chain source currently includes independent homotypic H2
coupling. Its original Q, rate, A/B and parameters are retained. H1's explicit
coupling is absent because its effective spatial pooling is supplied by the new
double exponential. No new S efficacy or H1 spatial parameters are applied to H2.

## Background and downstream scope

The new diagnostic reuses the prior HC entry's declared 1000 td uniform native
background, annulus and 1% increment. It is a native-Td diagnostic, not a new
display-calibrated optical experiment. The input mask uses original fine IS
positions before the original member projection. The larger source outside the
dynamic graph remains an explicit held uniform reservoir, with the old
unsupported B-member deficit retained rather than normalized away.

The new native fixed point uses the existing local equilibrium solver. The
existing complete inner background initializer is reused with explicit new
native dependencies. The already completed BC physiology parameters, AC/AII/RGC
parameters and equations are unchanged; their states are newly constructed at
the declared diagnostic background. This background is deliberately distinct
from the historical full-chain optical background. A new parameter/state
identity prevents old checkpoint reuse. No prior physiological or spatial
qualification is inherited by the new state.

## Completed acceptance

After successful preparation, the authorized acceptance command was:

```powershell
& 'D:\anaconda\envs\snn_env\python.exe' -X utf8 -m tools.accept_hc_eccentric
```

The single acceptance group checks two distinct H1 spatial scales, source
identities, complete-row normalization, all-member projection, local B and
unchanged H2. It then runs the existing native equations for the declared
annulus and L/M/S/LM diagnostic inputs. The H2 spectral comparison retains the
actual result and reports `H2_SPECTRAL_CONSTRAINT_UNRESOLVED` when the specified
qualitative relation is absent.

After these checks, one 50 ms full-chain run uses the existing scheduler and
exports Cone release, H1/H2 voltage, BC membrane/release, AC/AII states and RGC
conductance/voltage/expected rates. It checks current ports, state domains,
timestamps, units, changed-identity rejection and checkpoint continuation.
The receipt records actual wall time and peak working set only when executed.

The initial run reached checkpoint writing and exposed a missing
`default_adopted=False` qualification field in the acceptance caller. The field
was supplied without changing the model or storage gate. The failed receipt is
retained as `CHECKPOINT_WRITE_FAILURE.json`; only the affected full-chain phase
was repeated, with the same model identity and saved local evidence.

The new background maximum native RHS was 2.251e-11; checkpoint continuation
matched exactly. H2 S and LM peak magnitudes were 0.00579342 and 0.176291 native
voltage units, respectively, so stronger-S selectivity was not accepted. Fine H1
rows retain 43.998–89.163% held exterior mass. Engineering acceptance remains
conditional on that boundary and does not establish central H1 spatial
physiology, full spatial convergence or RGC fitting readiness. Existing output
directories are retained; do not repeat either command over completed assets.
