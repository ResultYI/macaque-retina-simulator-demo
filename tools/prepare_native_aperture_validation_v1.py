"""Prepare a deterministic subset of an existing optical fixture, never model data."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT / 'reports/isetbio_native_aperture_sampling_20261102_v1'
if any((OUT / name).exists() for name in ('spatial.json', 'uniform.json', 'VALIDATION.json')):
    raise FileExistsError('Use a fresh output directory; preserve previous validation')
OUT.mkdir(parents=True, exist_ok=True)
FIXTURE = ROOT / 'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
original_config = FIXTURE / 'tiles/tile_0_0/frames_000_015/export.json'
config = json.loads(original_config.read_text(encoding='utf-8'))
geometry_path = Path(config['geometry_mat'])
raw = loadmat(geometry_path)
sys.path.insert(0,str(ROOT))
from retina_sim.native_cone_apertures import native_apertures,PROFILE
from retina_sim.contracts import canonical_hash
from scipy.spatial import cKDTree
support=np.load(ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/rf.npy')
native_diameters,_=native_apertures(support)
_,support_rows=cKDTree(support).query(raw['is_mm'])
assert np.array_equal(support[support_rows],raw['is_mm'])
raw['reference_diameter_um']=native_diameters[support_rows]
raw['zone_diameter_um']=native_diameters[support_rows]
types = raw['types'].ravel()
indices = np.unique(np.concatenate((np.linspace(0, len(types)-1, 128, dtype=int),
    [np.flatnonzero(types == kind)[0] for kind in (1, 2, 3)])))
selected = dict(is_mm=raw['is_mm'][indices], types=types[indices],
    reference_diameter_um=raw['reference_diameter_um'].ravel()[indices],
    zone_diameter_um=raw['zone_diameter_um'].ravel()[indices])
savemat(OUT / 'geometry_subset.mat', selected, do_compression=True)
uniform = np.full((360, 640, 3, 1), round(config['reference_rgb'][0]*255), dtype=np.uint8)
savemat(OUT / 'uniform_rgb.mat', dict(source_rgb=uniform), do_compression=True)
source = dict(geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
    configuration_sha256=hashlib.sha256(original_config.read_bytes()).hexdigest(),
    selection='128 evenly spaced original rows plus first existing L/M/S, before response inspection',
    selected_original_rows_zero_based=indices.tolist(), count=len(indices),
    scopes=['uniform RGB through cached native OI and direct native oiCompute control',
            'existing first nonuniform source frame through native aperture capture and direct oiCompute'],
    qualification='bounded numerical equivalence fixture, not a new scientific stimulus or network run')
for scenario in ('uniform', 'spatial'):
    candidate = dict(config)
    candidate.update(aperture_model=PROFILE['profile_id'],aperture_profile_sha256=canonical_hash(PROFILE),geometry_mat=str(OUT / 'geometry_subset.mat'), frame_indices=[1],
        window_edges_seconds=[0., 1/30], native_control=True)
    if scenario == 'uniform':
        candidate['source_bank'] = str(OUT / 'uniform_rgb.mat')
    identity = hashlib.sha256(json.dumps(dict(source=source, scenario=scenario), sort_keys=True).encode()).hexdigest()
    candidate.update(request_signature=identity, tile_signature=identity)
    (OUT / f'{scenario}.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
(OUT / 'FIXTURE_IDENTITY.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
print(json.dumps(dict(fixtures_prepared=True, cones=len(indices), type_counts={str(t):int(sum(types[indices]==t)) for t in (1,2,3)})))
