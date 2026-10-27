"""Source construction and cached-input binding for one approved short A/B."""
from pathlib import Path
import json
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest, verified_path
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.group_parameter_binding import build_group_construction_seed
from retina_sim.group_time_storage import save_bundle
from retina_platform.config import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT) or out.exists() or not out.parent.is_dir():
        raise ContractError('Fresh local candidate preparation directory required')
    out.mkdir()
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    started = time.perf_counter()
    guard = MemoryGuard(16*1024**3, 4*1024**3)
    stage = 'STATIC_IDENTITY'
    def check():
        if (out/'REQUEST_STOP').exists() or time.perf_counter()-started > 600:
            raise ContractError('Candidate source preparation stopped; no warmup or fallback input')
        if not guard.check():
            raise ContractError('Candidate preparation memory guard: '+str(guard.stop_reason))
    try:
        check()
        approval_file = ROOT/'reports/group_time_candidate_20261025_v1/approval.json'
        approval = json.loads(approval_file.read_text(encoding='utf8'))
        if approval.get('answer') != '批准候选后端及一次短段验收':
            raise ContractError('Direct B-candidate and one-short-pair authorization required')
        a = GroupModelAssembly(ROOT/approval['group_artifact'])
        if digest(a.root/'manifest.json') != approval['group_manifest_sha256']:
            raise ContractError('Approved group artifact differs')
        source = ROOT/a.manifest['configuration']['source_run']
        policy_file = source/'input_policy.json'
        binding = dict(path=policy_file.relative_to(ROOT).as_posix(), sha256=digest(policy_file))
        # Preserve original frame chronology and global clock phase. The new
        # IVP starts on the 1ms grid, just before frames749(background)/750(video).
        protocol = dict(schema='retipath-group-short-paired-protocol-1', start_seconds=4993*.001,
            checkpoint_seconds=4997*.001, stop_seconds=5001*.001, input_origin_seconds=0.,
            input_frame_indices=[748,749,750], input_fps=150., clock_origin_seconds=0.,
            rod_step_seconds=.0001, BC_RGC_period_seconds=.001, monitor_step_seconds=.0001,
            recording_step_seconds=.0005, native_observation_delay_seconds=.00282,
            reference=dict(method='joint_DOP853',rtol=1e-6,atol=1e-9,max_step=.0001),
            candidate=dict(rtol=1e-6,atol=1e-9,fast_max_step=.0001,degree=7,slow_initial=.0005,slow_max=.001,max_corrections=3),
            numerical_budget=dict(path='configurations/numerical_equivalence/resident_native_fp64_v1.json',
                sha256='3cfab9768a856c13354ab38cc65b506da60661fe3e3b0f05e46c647c08981a77'),
            initialization='Original source-construction IVP on this group graph; no warmup/background qualification',
            qualification=dict(source_construction_legal=True,coupled_background_qualified=False,fit_eligible=False,default_adopted=False),
            approval_sha256=digest(approval_file),group_manifest_sha256=digest(a.root/'manifest.json'),
            physical_scope='All frozen0–6groupoutputs andcompleteupstream, no state/recording pruning',
            runtime_limits=dict(max_rss_bytes=16*1024**3,minimum_available_bytes=4*1024**3,max_whole_GPU_bytes=12*1024**3,
                total_preparation_wall_seconds=600,paired_test_wall_seconds=180))
        write('protocol.json', protocol)
        stage = 'SOURCE_CONSTRUCTION_INITIALIZATION'
        write('status.json',dict(stage=stage, model_forward_started=False))
        seed = build_group_construction_seed(a, ROOT, input_policy_binding=binding,
            rng_seed=RunConfig.project_default(ROOT).seed,time_seconds=protocol['start_seconds'],guard=check)
        initialized = time.perf_counter()
        stage = 'CACHED_STIMULUS_PROJECTION'
        check()
        policy = json.loads(policy_file.read_text(encoding='utf8'))
        folder = source/'centered_inputs_v2'
        for name in ('source_unique_gray8.npy','source_frame_to_unique.npy'):
            if digest(folder/name) != a.manifest['input']['source_files'][name]:
                raise ContractError('Cached stimulus source differs: '+name)
        frames = np.load(folder/'source_unique_gray8.npy',allow_pickle=False,mmap_mode='r')
        frame_map = np.load(folder/'source_frame_to_unique.npy',allow_pickle=False)
        native = a.array('input/group_native_td.npy')
        native_rows, rod_rows = {}, {}
        rod = a.populations['rods']
        record = a.manifest['populations']['rods']['coordinate']
        coords = np.load(verified_path(ROOT, record), allow_pickle=False, mmap_mode='r')
        rows = np.flatnonzero(rod.fine_to_group >= 0)
        xy = np.asarray(coords[rows], dtype=np.float64)
        groups = rod.fine_to_group[rows]
        from tools.prepare_foveal_demo_input import scene_multiplier
        for frame_index in protocol['input_frame_indices']:
            check()
            unique = int(frame_map[frame_index])
            native_rows[frame_index] = np.array(native[unique],copy=True)
            original_rod = scene_multiplier(frames[unique], xy, policy)*policy['rod']['proposed_reference_rate_Rstar_per_s']
            rod_rows[frame_index] = np.bincount(groups, weights=original_rod, minlength=rod.group_count)/rod.counts
        identity = dict(artifact_sha256=a.artifact_sha256,seed=seed.identity,
                        protocol_sha256=digest(out/'protocol.json'),input_policy=binding)
        bundle = save_bundle(out/'construction_seed',kind='source_construction_seed',identity=identity,
            payload=dict(states=seed.states,spike_states=seed.spike_states,inner_parameters=seed.inner_parameters,
                observation_prehistory=seed.observation_prehistory,input_reference=seed.input_reference,
                native_inputs=native_rows,rod_inputs=rod_rows,time_seconds=seed.time_seconds),
            qualification=protocol['qualification'])
        check()
        write('receipt.json',dict(status='LEGAL_SOURCE_CONSTRUCTION_IVP_READY',bundle=bundle,
            source_construction_seconds=initialized-started,cache_projection_and_write_seconds=time.perf_counter()-initialized,
            wall_seconds=time.perf_counter()-started,resources=guard.last,model_forward_started=False,
            qualified_background=False,optics_recomputed=False,source_binding=seed.identity))
        print(json.dumps(dict(output=str(out),status='LEGAL_SOURCE_CONSTRUCTION_IVP_READY',seconds=time.perf_counter()-started)))
    except Exception as exc:
        write('failure.json',dict(stage=stage,error=type(exc).__name__,message=str(exc),
            model_forward_started=False,elapsed_seconds=time.perf_counter()-started,resources=guard.last))
        raise


if __name__ == '__main__':
    main()
