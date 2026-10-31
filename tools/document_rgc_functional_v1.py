"""Write provenance and reproducibility notes from saved results and local pins."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    target = out / 'PARAMETER_ORIGIN_LEDGER.json'
    if target.exists():
        raise FileExistsError('Preserve parameter provenance record')
    params = json.loads((out / 'PARAMETERS.json').read_bytes())
    main_cfg = json.loads((ROOT / 'configurations/retina_platform_v1.json').read_bytes())
    rows = []
    for pop, actual in params['populations'].items():
        binding = main_cfg['rgc_output_defaults']['accepted_types'][pop]
        recipe_path = ROOT / binding['recipe_path']
        acceptance_path = ROOT / binding['acceptance']['path']
        recipe = json.loads(recipe_path.read_bytes())
        acceptance = json.loads(acceptance_path.read_bytes())
        matches = all(actual[k] == row['value'] for k, row in recipe['changes'].items())
        pinned = (hashlib.sha256(recipe_path.read_bytes()).hexdigest() == binding['recipe_sha256']
                  and hashlib.sha256(acceptance_path.read_bytes()).hexdigest() == binding['acceptance']['sha256'])
        if not matches or not pinned:
            raise ValueError('Saved actual parameters differ from documented local source: ' + pop)
        rows.append(dict(type=pop, recipe_path=binding['recipe_path'],
            acceptance_path=binding['acceptance']['path'], source=recipe.get('source'),
            output_values=recipe['changes'], actual_values_match_recipe=matches,
            local_recipe_and_acceptance_pins_match=pinned,
            evidence_grade=acceptance.get('evidence_grade', 'Source-limited ON-parasol engineering mapping'),
            inherited_scope=acceptance.get('scope', recipe.get('applicability')),
            inherited_limitations=acceptance.get('limitations'),
            new_subtype_fit=False, subtype_physiological_calibration=False))
    record = dict(
        scientific_parameters_modified=False, target_PSTH_used=False, new_Cell6_readout_used=False,
        membrane_profile='Inherited effective type profiles; actual per-member C60pF, gL2nS; no eccentricity fit',
        common_output_values='alpha90Hz, mu-53mV, beta1.67mV in all five frozen types',
        distinction='Four existing D cross-type engineering transfers remain D; this round does not promote their physiological evidence',
        physical_group_scaling='C and gL multiply member count once; grouped gE/gI already sum physical inputs once; currents in A',
        event_scaling='Rates and Bernoulli draws per sampled group object, never multiplied into independent fine-cell spike trains',
        source_binding_identity=params['source_identity'], types=rows)
    target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    notes = '''# RGC functional v1: reproducibility and limits

Use the project Python `.venv/Scripts/python.exe` from this project root.
Run into a new output directory; never rerun into an existing result directory.

```powershell
& '.\\.venv\\Scripts\\python.exe' -B tools/run_rgc_functional_v1.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW'
& '.\\.venv\\Scripts\\python.exe' -B tools/run_rgc_s_type_probe.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW'
& '.\\.venv\\Scripts\\python.exe' -B tools/analyze_rgc_functional_v1.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW'
& '.\\.venv\\Scripts\\python.exe' -B tools/run_rgc_coupled_qualification.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW'
& '.\\.venv\\Scripts\\python.exe' -B tools/analyze_rgc_functional_v1.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW' --coupled
& '.\\.venv\\Scripts\\python.exe' -B tools/document_rgc_functional_v1.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW'
```

Plotting is data-only. The model venv has no matplotlib; the existing base
`D:/anaconda/envs/retina_rf_snn/python.exe` has matplotlib3.11.0 and was used only
for figures, without importing or constructing the model:

```powershell
& 'D:\\anaconda\\envs\\retina_rf_snn\\python.exe' -B tools/plot_rgc_functional_v1.py --output 'reports/layerwise_physiology/RGC_functional_v1_NEW' --coupled
```

The original migration receipt and all historical ledgers remain unchanged.
The fresh resource ledger references the previous3185.5267654s plus measured
laptop subprocess wall time; each model subprocess retains a900s timeout.
The old08:33:45UTC deadline is archival and is not silently extended or erased.
This session follows the later explicit user authorization.

The first phase qualifies an explicit1ms BC-cache driven equation. Its tight
reference passed, but its saved full-inner endpoint rate difference exceeded
the stricter source allocation. That record remains in RESULT.json and
SOURCE_ENDPOINT_COMPARISON.csv; it is not whole-inner equivalence evidence.
The subsequent source_coupled_v1 phase integrates all6305 original inner states,
including reciprocal electrical coupling, under the original frozen Cone/HC
background and source BC protocol. It records all five RGC types and native
events, with its own standard/tight pair and frozen-source endpoint comparison.
Neither phase is a complete native/global retinal checkpoint.

The five types use existing source-bound effective parameters. In particular,
the common90Hz/-53mV/1.67mV output mapping is inherited. Four previous subtype
adoptions remain D cross-type engineering transfers; no new Cell6 normalization,
PSTH-derived physical gain, subtype fit or parameter optimization occurred.
See PARAMETERS.json and PARAMETER_ORIGIN_LEDGER.json for exact local source pins.

Main ON/OFF polarity uses native Cone release decrease/increase, not a new
optical contrast calibration. The S-bistratified type has an OFF-like mixed
response to common input and a positive response to the selective S_ON_BC
terminal probe. This does not validate optical color opponency. The constant
background, gI-removal and selective-S controls use explicitly driven original
equations; no control changed saved source parameters, topology or production.

Expected source-node output retains the original probability-law-after-tick
convention. Each900-bin packet separately saves the physically relevant
pre-tick expected binary count rate, actual binary draws and native within-bin
first-event times. There is no sampled future bin at the901st endpoint.
Member-weighted plots are descriptive group means. Actual events are one
sampled group object each; no fine-cell spike trains are replicated.

Scientific limits remain: no whole-retina background qualification, no0–6deg
calibration, no subtype-specific physiological fit, no general image/video
validation, no intrinsic ordinary-AC adaptation claim, and no action-potential
reset/refractory mechanism beyond the retained three-bin history sampler.
FIT_READY=false, background_qualified=false, production_adopted=false.

Next single step after this layer closes: return to the already agreed
cross-eccentricity calibration stage, preserving these evidence limitations.
'''
    (out / 'REPRODUCE.md').write_text(notes, encoding='utf8')
    print(json.dumps(dict(parameter_origin_types=len(rows), no_new_fit=True)))


if __name__ == '__main__':
    main()
