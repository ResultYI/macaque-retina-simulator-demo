# Functional Cone–H1 spatial mode

The new eccentricity-dependent implementation is documented in
[cone_hc_eccentric.md](cone_hc_eccentric.md). It is an explicit configuration and
is currently written but not run or tested. This page describes the retained
constant-scale entry and its historical acceptance.

For a new supported local CPU research task:

```python
from retina_sim.assembly import RetinaBuilder
import numpy as np

model = RetinaBuilder.build_research_cone_hc(repo_root)
runner = model.runner()
trace = runner.advance(0.12, np.arange(61) * 0.002)
```

The default at this research entry point is `configurations/research/cone_hc_biexponential_functional.json`. Explicit historical configurations remain unchanged, including `cone_hc_uniform.json` and the earlier contact-masked `cone_hc_biexponential.json`. The ordinary production Builder and frozen historical tasks retain their existing selection. Unsupported research execution modes raise `ContractError`; they do not select uniform automatically.

Both functional components use existing finite OPL support (radius3.526911366748626mm), only L/M cones, and separate full-source normalization:

`A = 0.15 * exp(-distance_mm/0.020)/Zs + 0.85 * exp(-distance_mm/0.300)/Zl`.

The short component has no anatomical contact mask or contact-strength multiplier. Effective inputs do not add anatomical synapses. B remains the original local contact operator; H2 remains its independent uniform model. All van Hateren2005 time parameters, state units, and downstream equations remain unchanged.

`HC_biexponential_functional.json` defines the operator. `prepare_cone_hc_functional.py` reuses the existing complete typed-source envelope, streams one neighborhood at a time, and saves CSR dynamic matrices plus exact held mass. No full249-by635663 dense operator is allocated. The Builder verifies the asset receipt and loads fixed matrices; the CPU RHS only uses existing sparse products and temporal dynamics.

The initial asset is the existing321-Cone/249-H1 local research geometry around H1 parent2891. Its external field uses the previously explicit held-background approximation, and only registered Cone166128 may lack local feedback. The strict closure gate is still independent. This asset does not imply arbitrary geometry, GPU, full-video, or whole-parent support.

Readouts include H1 native voltage, all/local Cone release, pooled release, core feedback, `Vbar_core`, `aI_core`, `pooled_Vbar_target`, and `pooled_aI_target`. Voltage remains `vH_native_voltage`, without a physicalmV calibration; release is effective dimensionless drive.

Checkpoint schema/state layout is unchanged. New functional weights have their own model/native identities, so checkpoints cannot cross kernel definitions. Explicit frozen legacy configurations can restore the recorded previous implementation's checkpoint identity only after the same configurations, matrices, parameters, source and state layout are revalidated.

Fine-member release perturbations enter the existing all-member group mean and one-way Cone→BC→AC→RGC port through `FineMemberReleaseProjection` and `ConeHCGroupPortRunner`. The parent's auxiliary Cone/HC states retain their own identity. The existing downstream time scheduler and checkpoint format are reused.

The historical finite-domain spatial-convergence failure is retained. The final functional kernel has not received spatial physiological qualification; this engineering acceptance does not confer it. Do not infer a pure Cone surround or calibratedH1/RGC prediction from this diagnostic.
