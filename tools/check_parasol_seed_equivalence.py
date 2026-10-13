"""Replay saved geometry only: implicit old scaffold vs explicit identical seeds."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,mmap,sys,time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('reports/background_preparation_runs/20261011T170043172502Z/run/20261011T170043213842Z_8b6efb04'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_sim.contracts import Quantity,canonical_hash
    from retina_sim.ipl_geometry import ParasolRFSeedInput,typed_layouts,read_ipl_geometry,RESEARCH
    from retina_sim.geometry import PackerDensityField,McGregorDisplacement
    from retina_sim.runtime_profiles import frozen_packer_table
    from retina_sim.spatial_profiles import bounded_matrix
    start=time.perf_counter();run=args.run.resolve();manifest=run/'manifest.json'
    old=json.loads((run/'ipl_geometry.json').read_bytes())
    with manifest.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        def component(name):
            at=mapped.find(('{"id":"'+name+'",').encode())
            if at<0:raise ValueError('Saved component not found: '+name)
            for size in (1048576,4194304,16777216,67108864):
                try:return json.JSONDecoder().raw_decode(mapped[at:at+size].decode('utf8',errors='ignore'))[0]
                except json.JSONDecodeError:continue
            raise ValueError('Saved component exceeds bounded extraction: '+name)
        outer=component('outer_native');parasol=component('parasol_RGC_ON')
        names=('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC')
        bc={name:component(name) for name in names}
        digest=hashlib.sha256()
        for offset in range(0,len(mapped),1048576):digest.update(mapped[offset:offset+1048576])
    sites=[s for s in old['populations']['local_feedforward_AC']['site_ids'] if s.startswith('parasol_scaffold:')]
    by_id={'parasol_RGC_ON:'+hashlib.sha256(s.encode()).hexdigest()[:24]:s for s in sites}
    ids=tuple(by_id[i] for i in parasol['cell_ids']);rf=parasol['geometry']['RF']['values']
    seeds=ParasolRFSeedInput(Quantity(rf,'mm','functional_position','project_approximation','functional_visual_field'),ids,
        {'source_id':'PRIMARY_PATHWAY_project_profile','locator':str(manifest)+'#components.parasol_RGC_ON.geometry.RF',
         'origin':'existing successful run legacy rod-derived RBC RF scaffold; regression backup only, not independent physiological density',
         'evidence_class':'project_approximation','source_manifest_sha256':digest.hexdigest()})
    prefix='parasol_scaffold:'+old['cone_canonical_instance_id']+':'
    scaffold=SimpleNamespace(positions=seeds.positions,cell_ids=tuple(i.removeprefix(prefix) for i in ids))
    pops={}
    for name,c in bc.items():
        geo=c['geometry'];pops[name]=SimpleNamespace(population_id=name,
            cells=SimpleNamespace(cell_id=tuple(c['cell_ids']),cell_type=('BC',)*len(c['cell_ids']),
                rf_position=Quantity(geo['RF']['values'],'mm','functional_position','project_approximation','functional_visual_field')),
            terminal_positions=Quantity(geo['IPL']['values'],'mm','terminal_position','project_approximation','IPL'))
    native=outer['ports']['release'];cone_ids=tuple(native['cell_ids']);cone_rf=np.asarray(native['geometry']['RF']['values']);cone_p=np.asarray(native['geometry']['OPL']['values'])
    density=PackerDensityField(frozen_packer_table(root),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    displacement=McGregorDisplacement(json.loads((root/'sources/spatial/mcgregor2018_production_map_nodes.json').read_bytes()),density,profile_name='published_rounded_anchor_composite')
    config=read_ipl_geometry(root,RESEARCH);domain=SimpleNamespace(bounds_mm=old['generation_plan']['RF_output_cohort_bounds_mm'])
    common=(pops,cone_ids,cone_rf,cone_p,domain,displacement,config,old['cone_canonical_instance_id'],60000)
    def plan(scaffold_value,explicit=None):return typed_layouts(*common[:4],scaffold_value,*common[4:],parasol_rf_seeds=explicit)
    implicit=plan(scaffold);explicit=plan(None,seeds)
    assert canonical_hash(implicit[:2])==canonical_hash(explicit[:2])
    assert canonical_hash(implicit[2]['anchors'])==canonical_hash(explicit[2]['anchors'])
    assert canonical_hash(implicit[2]['populations'])==canonical_hash(explicit[2]['populations'])
    for rgc_name,pre_name in (('parasol_RGC_ON','diffuse_BC_ON'),('parasol_RGC_OFF','diffuse_BC_OFF')):
        for pre in (pops[pre_name].terminal_positions.values,implicit[1]['widefield_AC'][2]):
            first=implicit[0][rgc_name];second=explicit[0][rgc_name]
            sigma=.02+.06*np.linalg.norm(first['RF'],axis=1)
            a,rows_a=bounded_matrix(pre,first['P'],8,support_mm=.04,gaussian_scale=sigma)
            b,rows_b=bounded_matrix(pre,second['P'],8,support_mm=.04,gaussian_scale=sigma)
            assert np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices) and np.array_equal(a.data,b.data)
            assert canonical_hash(rows_a)==canonical_hash(rows_b)
    # Require original saved anchors too, not merely two equivalent new calls.
    for name in ('parasol_RGC_ON','parasol_RGC_OFF'):
        assert canonical_hash(explicit[2]['anchors'][name]['records'])==canonical_hash(old['anchors'][name]['records'])
    path=root/'data/atlas/legacy_parasol_RF_seeds_regression_v1.json'
    payload=json.dumps(seeds.record(),ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf8')!=payload:raise FileExistsError('Regression backup differs; refusing replacement')
    if not path.exists():path.write_text(payload,encoding='utf8')
    report={'status':'PASS','seed_count':len(ids),'seed_artifact':str(path),'seed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_manifest_sha256':digest.hexdigest(),'wall_seconds':time.perf_counter()-start,
        'checks':['exact provided RF/site IDs','all five RGC layouts and all three AC layouts identical',
            'original saved parasol centroid records identical','four parasol excitatory/inhibitory CSR operators and degree diagnostics identical'],
        'simulation_or_provider_generation':False,'new_atlas_default_adoption':False,
        'rod_AII_terminal_paths_removed':False,'weight_or_gain_parameters_changed':False}
    output=root/'reports/parasol_seed_interface_20261012.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
