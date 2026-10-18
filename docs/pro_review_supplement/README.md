# WIP physiology and performance review

This is the current review snapshot, not final three-ROI acceptance. `review_status.json` is the concise status and performance entry point. The current default is unchanged; all automatic final publishers were stopped for this explicit WIP push.

`transition_original_HC_response.png` shows the old anatomical-HC transition baseline: a20ms static image versus background, after2.1s required numerical preparation. It is not a result for the new functional HC operator. Raw numeric response arrays retain absolute times, original units and byte-exact values. Finite outputs plus nonzero native/BC/RGC differences establish limited numerical pathway operation, not physiological amplitude, latency or spike validation.

The functional HC alternative keeps existing HC coordinates/densities and uses source divergence anchors with explicit project transfer/allocation/normalization choices. Its zero functional owner gaps do not repair the7770 anatomical-reference gaps. S/H2 kinetics, many inner dynamics and subtype completeness remain uncalibrated. The central empty-owner shape bug is repaired; reassembly exposed AII chemical excitation support missing for some cells. That condition remains blocked for review.

Performance is an active blocker:2.1s of simulated background took about4.3h; only about21% was within the integrator. Residual checks/evaluation/metadata/conversion/checkpoint costs still need separate short profiling. Native CUDA speedups must not be presented as whole-network speedups.

Large generated populations, full CSR arrays, complete checkpoints, source papers, vendor MATLAB/ISET code and machine profiles are excluded. `review_index.json` records exact omitted-file identities. The official human cone source is https://purl.stanford.edu/vm447vf0975 (right/nominal-fov-58deg.mat, SHA25654eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728). Use the pinned ISET versions in `review_status.json`; inspect `tools/build_human_disk3mm.py --help`, `tools/build_functional_hc.py --help` and `tools/build_disk3mm.py --help`. Copy the example optical profile to its ignored local name and set installed paths. Exact cold reproduction requires the retained hash-bound master assets; any regeneration creates a new identity and requires its own checks.

The optional float64 GPU DOP853 port follows the installed SciPy implementation/coefficients. SciPy BSD license and bundled notices are retained in SCIPY_LICENSE.txt; no vendor code, MAT population, source paper or MATLAB distribution is included.
