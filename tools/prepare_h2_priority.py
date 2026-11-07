"""Prepare sparse complete-source H2 weights, preserving all existing H1 assets."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import traceback
import argparse
import numpy as np
from scipy import sparse

from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_eccentric import checked_path, digest
from retina_sim.hc_group_projection import _normalize_complete_rows
from retina_sim.h2_priority import complete_priority, DEFAULT_CONFIGURATION
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.vh2005 import SOURCE_VALUES
from tools.prepare_hc_eccentric import write
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-failed',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    config_path=checked_path(ROOT,DEFAULT_CONFIGURATION)
    cfg=json.loads(config_path.read_bytes())
    out=(ROOT/cfg['assets_directory']).resolve()
    if not out.is_relative_to(ROOT):raise ContractError('Preparation output must stay in workspace')
    if args.resume_failed:
        previous=json.loads((out/'RECEIPT.json').read_bytes())
        if previous.get('status')!='FAILED_RETAINED':raise ContractError('Only a failed preparation may resume')
        with (out/'PREPARATION_FAILURE.json').open('xb') as stream:stream.write((out/'RECEIPT.json').read_bytes())
    else:
        out.mkdir(exist_ok=False)
    result=dict(status='PREPARING',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        base=RetinaBuilder.build_research_cone_hc(ROOT,cfg['base_configuration'])
        prior=json.loads((ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/eccentric_acceptance/RESULT.json').read_bytes())
        assert base.identity==prior['model_identity']
        spec_path=checked_path(ROOT,cfg['H2_configuration']);spec=json.loads(spec_path.read_bytes())
        native=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
        source_path=native/'NATIVE_ASSEMBLY_RESULT.json';source=json.loads(source_path.read_bytes())
        cones=FrozenDiskTable(ROOT,source['cone_table']);ct=cones.extras['subtype'].astype('U1')
        raw_path=native/'RAW_HC_BY_CONE.npz';raw=sparse.load_npz(raw_path).tocsr()
        m=base.members;fine_rows=np.flatnonzero(m['hc_types']=='H2')
        h2raw=raw[m['hc_parent_rows'][fine_rows]].tocsr()
        newfull=complete_priority(h2raw,ct,spec);oldfull,_=_normalize_complete_rows(h2raw)
        hg=m['hc_group'][fine_rows]
        projection=sparse.csr_matrix((1/m['hc_counts'][hg],(hg,np.arange(len(hg)))),
                                     shape=(base.A.shape[0],len(hg)))
        new_group_full=(projection@newfull).tocsr();old_group_full=(projection@oldfull).tocsr()
        ci=m['cone_parent_rows'];cm=m['cone_group']
        cp=sparse.csr_matrix((np.ones(len(ci)),(np.arange(len(ci)),cm)),shape=(len(ci),base.A.shape[1]))
        new_dynamic=(new_group_full[:,ci]@cp).tocsr()
        old_dynamic=(old_group_full[:,ci]@cp).tocsr()
        np.testing.assert_allclose((old_dynamic[base.h2_rows]-base.A[base.h2_rows]).data,0.,rtol=0,atol=2e-14)
        mask=np.zeros(base.A.shape[0]);mask[base.h1_rows]=1.
        A=(sparse.diags(mask)@base.A+new_dynamic).tocsr()
        exterior=np.ones(len(ct),bool);exterior[ci]=False
        held=base.mass['A_group_held'].copy()
        held[base.h2_rows]=np.asarray(new_group_full[base.h2_rows][:,exterior].sum(1)).ravel()
        selected_row=cfg['representative_H2_group']
        all_sources=new_group_full[selected_row].indices
        missing=all_sources[exterior[all_sources]]
        if len(missing)!=2 or np.any(ct[missing]!='S') or np.sum(ct[all_sources]=='S')!=3:
            raise ContractError('Expected audited H2 with exactly two exterior and one dynamic S Cone')
        selected_mass=np.asarray(new_group_full[:,missing].sum(1)).ravel()
        old_selected_mass=np.asarray(old_group_full[:,missing].sum(1)).ravel()
        if np.any(selected_mass[base.h1_rows]):raise ContractError('H1 cannot receive selected S exterior sources')
        if np.any(selected_mass>held+2e-14):raise ContractError('Selected exterior exceeds held support')
        np.testing.assert_allclose(np.asarray(A.sum(1)).ravel()+held,1.,atol=2e-14,rtol=0)
        # The existing exterior interface has one common 1000 Td state and fixed
        # feedback; verify the background reservoir used by this exact base.
        active=base.mass['A_group_held']>0
        released=base.native.boundary['A_release'][active]/base.mass['A_group_held'][active]
        vbars=base.native.boundary['A_Vbar'][active]/base.mass['A_group_held'][active]
        np.testing.assert_allclose(released,released[0],atol=1e-14,rtol=0)
        np.testing.assert_allclose(vbars,vbars[0],atol=1e-14,rtol=0)
        feedback=float(released[0]*SOURCE_VALUES['g_t'])
        for name,value in [('A_group',A),('H2_A_fine_complete',newfull),('H2_uniform_fine_complete',oldfull),
                           ('H2_A_group_complete',new_group_full),('H2_uniform_group_complete',old_group_full)]:
            sparse.save_npz(out/(name+'.npz'),value)
        np.savez_compressed(out/'MASS_LEDGER.npz',A_group_held=held,
            selected_exterior_group_mass=selected_mass,uniform_selected_exterior_group_mass=old_selected_mass,
            fixed_unstimulated_group_mass=held-selected_mass,
            H2_fine_parent_rows=m['hc_parent_rows'][fine_rows],H2_group_rows=base.h2_rows,
            fine_group=hg,all_cone_types=ct,dynamic_cone_parent_rows=ci,
            A_fine_held=np.asarray(newfull[:,exterior].sum(1)).ravel())
        np.savez_compressed(out/'EXTERIOR_MEMBERS.npz',cone_parent_rows=missing,
            cone_keys=cones.keys[missing].astype('U'),cone_types=ct[missing],cone_pedicle_mm=cones.coordinates['opl_pedicle'][missing],
            representative_all_source_rows=all_sources,representative_source_types=ct[all_sources])
        sources=[config_path,spec_path,source_path,raw_path,Path(__file__),ROOT/'retina_sim/h2_priority.py',
            ROOT/'retina_sim/hc_research_native.py',ROOT/'retina_sim/hc_timevarying_exterior.py',
            ROOT/cfg['base_configuration'],ROOT/base.config['assets_directory']/'SPATIAL_RECEIPT.json']
        identity=canonical_hash(dict(specification=spec,raw=matrix_id(h2raw),full=matrix_id(newfull),
            fine_parent_rows=m['hc_parent_rows'][fine_rows],source_types=ct,
            member_projection_sha256=digest(ROOT/base.config['assets_directory']/'MEMBERS.npz')))
        result.update(status='PRECOMPUTED',uniform_model_identity=base.identity,H2_weight_identity=identity,
            specification=spec,A_identity=matrix_id(A),B_identity=matrix_id(base.B),H1_identity=matrix_id(A[base.h1_rows]),
            H2_Q_identity=matrix_id(base.native.Q['H2']),H2_parameters=dict(base.native.parameters),
            H1_calibration=base.calibration,geometry_regenerated=False,contacts_added=0,
            preserved_raw_values=True,crop_renormalized=False,
            exterior_feedback_native=feedback,exterior_release_background=float(released[0]),
            exterior_Vbar_background=float(vbars[0]),selected_exterior_parent_rows=missing,
            exterior_policy='Existing FixedFeedbackCone: selected two actual S members receive native light modulation; exterior HC feedback remains fixed, not a complete peripheral HC network',
            source_files={p.relative_to(ROOT).as_posix():digest(p) for p in sources},
            output_files={p.name:digest(p) for p in out.glob('*.npz')})
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(wall_seconds=time.perf_counter()-start,peak_working_set_bytes=peak_memory_bytes(),
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(out/'RECEIPT.json',result)
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','peak_working_set_bytes')}),flush=True)


if __name__=='__main__':main()
