"""ISETBio aperture generation; Python only partitions/caches complete support."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import uuid

import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree

from .contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {'schema': 'isetbio-native-cone-aperture-1',
           'profile_id': 'isetbio_native_pillbox_v1', 'shape': 'Pillbox',
           'engine': 'cMosaic.computeConeApertures',
           'eccVaryingConeAperture': True, 'rodIntrusionAdjustedConeAperture': False,
           'smoothLocalVariations': False, 'custom_eccentricity_curve': False}


def native_apertures(positions_mm):
    xy = np.asarray(positions_mm, dtype=np.float64)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 7
            or not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy)):
        raise ContractError('Native apertures require >=7 unique finite complete-support positions')
    sources = ['tools/generate_native_cone_apertures.m', 'tools/native_cone_apertures_from_positions.m',
               'vendor_pinned/isetbio/cones/cmosaic/@cMosaic/computeConeApertures.m',
               'vendor_pinned/isetbio/cones/cmosaic/@cMosaic/cMosaic.m',
               'vendor_pinned/isetbio/cones/cmosaic/@cMosaic/computeApertureAreasMetersSquared.m',
               'vendor_pinned/isetbio/ganglioncells/+RGCmodels/+Watson/+convert/positionsToSpacings.m',
               'retina_sim/native_cone_apertures.py']
    pins = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    identity = hashlib.sha256(xy.tobytes() + json.dumps(pins, sort_keys=True).encode()).hexdigest()
    folder = ROOT / 'data/isetbio_native_aperture_cache' / identity
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / 'apertures.npz'
    if not result.exists():
        lock = folder / 'RUNNING'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ContractError('Native aperture generation already owned: ' + str(lock)) from exc
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        try:
            # Exact six-neighbour halos remove ROI/chunk boundary bias. No
            # spacing-to-diameter or eccentricity formula is evaluated here.
            tree = cKDTree(xy)
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            blocks = [order[i:i + 512] for i in range(0, len(xy), 512)]
            positions = np.empty((len(blocks), 1), object)
            query_rows = positions.copy(); local_rows = positions.copy()
            for b, rows in enumerate(blocks):
                _, neighbours = tree.query(xy[rows], k=7)
                support = np.unique(neighbours)
                positions[b, 0] = xy[support] * 1000
                query_rows[b, 0] = (rows + 1).astype(float)
                local_rows[b, 0] = (np.searchsorted(support, rows) + 1).astype(float)
            savemat(folder / 'input.mat', dict(positions=positions, query_rows=query_rows,
                    local_rows=local_rows, count=float(len(xy))), do_compression=True)
            profile = json.loads((ROOT / 'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
            quote = lambda p: str(p).replace("'", "''")
            expression = ("addpath('" + quote(ROOT / 'tools') + "');generate_native_cone_apertures('"
                          + quote(folder / 'input.mat') + "','" + quote(folder / 'output.mat') + "');")
            env = dict(os.environ, MATLAB_PREFDIR=str(folder / 'matlab_preferences'))
            Path(env['MATLAB_PREFDIR']).mkdir(exist_ok=True)
            with (folder / 'matlab.log').open('wb') as log:
                completed = subprocess.run([profile['matlab_executable'], '-batch', expression],
                    stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3600,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if completed.returncode:
                raise ContractError('Native ISETBio aperture generation failed: ' + str(folder / 'matlab.log'))
            raw = loadmat(folder / 'output.mat')
            diameters = raw['diameters'].ravel(); areas = raw['areas'].ravel()
            if (diameters.shape != (len(xy),) or not np.isfinite(diameters).all()
                    or not np.all(diameters > 0) or not np.isfinite(areas).all() or not np.all(areas > 0)):
                raise ContractError('Invalid native aperture output')
            temporary = folder / ('apertures_' + uuid.uuid4().hex + '.npz')
            np.savez_compressed(temporary, diameters_um=diameters, effective_area_m2=areas)
            (folder / 'receipt.json').write_text(json.dumps(dict(profile=PROFILE, sources=pins,
                support_sha256=hashlib.sha256(xy.tobytes()).hexdigest(), count=len(xy),
                normal_exit_code=completed.returncode, exact_six_neighbour_halos=True,
                aperture_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest()), indent=2), encoding='utf8')
            os.replace(temporary, result)
        finally:
            lock.unlink()
    receipt = json.loads((folder / 'receipt.json').read_bytes())
    if hashlib.sha256(result.read_bytes()).hexdigest() != receipt['aperture_sha256']:
        raise ContractError('Native aperture cache hash mismatch')
    with np.load(result, allow_pickle=False) as z:
        return z['diameters_um'].copy(), z['effective_area_m2'].copy()
