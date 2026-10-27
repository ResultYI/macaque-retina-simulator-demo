"""Lazy, read-only consumer of cell6-aligned-replay-1; never imports a solver."""
from pathlib import Path
import json

import numpy as np

from .replay_data import read_json, require, sha

PHASES=("primary","post")
INDEX_NAMES=("time_source_s","model_time_s","side","phase","event_generation",
             "clock_rod","clock_bc","clock_rgc","input_frame")


def primary_sides(model_times,frames,fps,origin):
    """Match original calendar's exact float keys, never an isclose/snapping rule."""
    edges={int(f)/fps for f in frames if origin<int(f)/fps<float(model_times[-1])}
    sides=np.asarray([b'left' if float(t) in edges else b'right' for t in model_times],dtype='S5')
    sides[0]=b'right'
    return sides


def discover(root,case_id):
    """Bounded manifest discovery only; never enumerate the large values folders."""
    folder=Path(root)/"reports/rgc_stage1_cpu_parallel_full_v1/workers"/case_id/"aligned"
    paths=[]
    for p in sorted(folder.glob("attempt_*/manifest.json")):
        status=p.parent/"STATUS.json"
        if status.exists() and read_json(status).get("status")=="COMPLETE_ALIGNED":paths.append(p)
    require(len(paths)<=1,"Multiple complete aligned products; choose the exact manifest explicitly")
    return paths[0] if paths else None


class AlignedProduct:
    def __init__(self,path,expected=None,*,allow_synthetic=False):
        self.path=Path(path).resolve()
        self.root=self.path.parent
        self.manifest=read_json(self.path)
        m=self.manifest
        self.synthetic=m.get("synthetic_fixture") is True or m.get('synthetic_test_only') is True
        require(not self.synthetic or allow_synthetic,"Synthetic fixture is not a real recording")
        allowed_status={'COMPLETE_ALIGNED','SYNTHETIC_MICROCHECK_COMPLETE'} if self.synthetic and allow_synthetic else {'COMPLETE_ALIGNED'}
        require(m.get("schema")=="cell6-aligned-replay-1" and m.get("status") in allowed_status,"Aligned product is not complete")
        require(m.get("model_or_solver_changed") is False and m.get("group_state_not_fine_cell_state") is True,"Recording qualification differs")
        status=read_json(self.root/"STATUS.json")
        require(status.get("status")==m['status'] and status.get("manifest_sha256")==sha(self.path),"Missing / incomplete / changed publication receipt")
        require(not (self.root/"OBSERVER_FAILURE.json").exists(),"Aligned observer recorded a failure")
        contract_path=self.root/"recording_contract.json"
        require(sha(contract_path)==m["recording_contract_sha256"],"Aligned recording contract bytes differ")
        self.contract=read_json(contract_path)
        c=self.contract
        require(c.get('schema')=='cell6-aligned-observer-contract-1',"Unsupported recording contract")
        require(not c.get('synthetic_test_only',False) or self.synthetic,"Synthetic recording contract cannot masquerade as real")
        self.n=int(c["source_nodes"]);self.dt=float(c["source_dt_s"])
        require(self.dt==.0001 and self.n>1,"Unsupported aligned time grid")
        require(self.synthetic or self.n==71001,"Real recording must contain all 71001 source nodes")
        require(c["product_schema"]==m["schema"],"Recording / product schema differs")
        provenance=m["provenance"]
        require(all(provenance.get(k)==v for k,v in c["provenance"].items()),"Recording / product provenance differs")
        require(c['case_id']==provenance['case_id'],"Recording case ID differs")
        if not self.synthetic:
            require(expected is not None,"Real aligned product requires the verified case binding")
            required=dict(case_id=expected.case_id,scientific_contract_sha256=expected.spec_sha,
                cache_identity=expected.caches[expected.case_id]["cache_identity"],
                source_endogenous_sha256=expected.caches[expected.case_id]["payload_sha256"],
                stimulus_sha256=expected.case["stimulus_sha256"],drive_sha256=expected.case["drive_sha256"],
                source_seed_sha256=expected.spec["source_seed_sha256"],target_global_id=expected.spec["target"]["global_id"])
            required['source_model_origin_s']=float(expected.drive_origin_s if hasattr(expected,'drive_origin_s') else expected.drive['model_origin_s'])
            require(all(provenance.get(k)==v for k,v in required.items()),"Aligned product / selected frozen case identity differs")
        self.case_id=provenance["case_id"]
        self.origin=float(provenance["source_model_origin_s"])
        require(self.origin==float(c["source_model_origin_s"]),"Model time origin differs")
        self.stop=round((self.n-1)*self.dt,10)
        self.streams={"primary":dict(shared_point_index=m["shared_point_index"],signals=m["signals"]),"post":m["post_commit_product"]}
        self.indices={};self.fields={};self._mapped={};self._ranges={};self._verified={};self._geometry_loaded={}
        for phase in PHASES:
            stream=self.streams[phase]
            if phase=="post":require(stream["rows"]==self.n,"Post stream is incomplete")
            indices={}
            for name in INDEX_NAMES:
                descriptor=stream["shared_point_index"][name]
                require(descriptor["rows_written"]==self.n,"Point index is an incomplete prefix")
                a=self.map_descriptor(descriptor)
                require(a.shape==(self.n,),"Shared index extent differs: "+name)
                indices[name]=a
            require(np.array_equal(indices["time_source_s"],np.arange(self.n)*self.dt),"Shared source grid differs")
            require(np.allclose(indices["model_time_s"],indices["time_source_s"]+self.origin,rtol=0,atol=1e-11),"Model/source clock bridge differs")
            require(np.isin(indices["side"],[b"left",b"right"]).all(),"Unknown observation side")
            phases=(b"initial-precommit",b"pre-commit") if phase=="primary" else (b"initial-postcommit",b"post-commit")
            require(indices["phase"][0]==phases[0] and (indices["phase"][1:]==phases[1]).all(),"Observation phase semantics differ")
            for name in INDEX_NAMES[4:]:
                require(indices[name].dtype.kind in "iu","Event/clock index must be integer")
            for name in ("event_generation","clock_rod","clock_bc","clock_rgc"):
                require((np.diff(indices[name])>=0).all(),"Event/clock indices move backwards")
            if not self.synthetic:
                require(np.isin(indices['input_frame'],expected.frames).all(),"Recorded input frame is absent from the frozen case drive")
            self.indices[phase]=indices
            fields={}
            for f in stream["signals"]:
                require(f["id"] not in fields,"Duplicate aligned field ID")
                require(f["point_index"]==phase and f["shape"]==[self.n,f["count"]] and np.dtype(f["dtype"])==np.float64,
                        "Field time / phase / shape / dtype contract differs")
                require(f["key"][0]==f["population"] and f["key"][-1]==f["field"],"Field key / population differs")
                self._validate_axis(f)
                fields[f["id"]]=f
            required_states={tuple(s["key"]):(s["count"],s["unit"]) for s in c["state_fields"]}
            actual_states={tuple(f["key"]):(f["count"],f["unit"]) for f in fields.values() if f["kind"]=="states"}
            require(actual_states==required_states,"The declared state blocks are missing or changed")
            self.fields[phase]=fields
        a,b=self.indices["primary"],self.indices["post"]
        require(np.array_equal(a["time_source_s"],b["time_source_s"]) and np.array_equal(a["model_time_s"],b["model_time_s"]),"Pre/post streams do not share exact source/model times")
        require((b["side"]==b"right").all(),"Original post observations must use the right side")
        require((b['event_generation']-a['event_generation']==1).all(),"Post-commit label lacks the actual event-generation increment")
        rod_delta=b['clock_rod']-a['clock_rod']
        require(rod_delta[0]==0 and (rod_delta[1:]==1).all(),"Post rod clock is not the actual canonical commit witness")
        bc_delta=b['clock_bc']-a['clock_bc'];rgc_delta=b['clock_rgc']-a['clock_rgc']
        require(bc_delta[0]==1 and rgc_delta[0]==1 and np.isin(bc_delta,[0,1]).all() and
                np.array_equal(bc_delta,rgc_delta),"BC/RGC commit witnesses differ from initial and native1ms semantics")
        if not self.synthetic:
            fps=expected.spec['numerical']['input_fps']
            # Original calendar keeps distinct floating-point keys. Near-grid
            # frame events are never snapped onto the canonical author row.
            sides=primary_sides(a['model_time_s'],expected.frames,fps,self.origin)
            require(np.array_equal(a['side'],sides),"Primary side differs from the frozen original RGC frame/calendar convention")
        require(set(self.fields["primary"])==set(self.fields["post"]),"Pre/post fields differ")
        for fid,f in self.fields["primary"].items():
            other=self.fields["post"][fid]
            require(all(f[k]==other[k] for k in ("kind","key","unit","count","axis")),"Pre/post field identity differs")
            require(all(f.get(k)==other.get(k) for k in ("group_ids","members","centroid_mm","coordinate_role","cell_count_not_applicable","state_semantics")),"Pre/post group geometry differs")
        check=m["legacy_reference_check"]
        require(check["passed"] is True and check["baseline1ms_expected_exact"] is True and
                check["RGC_ports"] and all(x["bitwise_equal"] is True for x in check["RGC_ports"]),"Canonical original RGC reference check did not pass")
        if not self.synthetic:
            expected_ports={(pop,field) for pop in expected.spec["recording"] for field in ("voltage","gE","gI")}
            require({(x["population"],x["field"]) for x in check["RGC_ports"]}==expected_ports,"Incomplete canonical RGC comparison receipt")
        self.geometry=read_json(self.checked_path(m["geometry"]))
        rate=m["rate_bin_product_separate"]
        require(rate["clock_s"]==.001 and rate["bins"]==c["rate_bins"]==(self.n-1)//10 and
                np.array_equal(rate["time_source_s"],np.arange(c["rate_bins"])*.001),"Separate 1ms rate clock differs")

    def checked_path(self,descriptor):
        relative=Path(descriptor["path"])
        p=(self.root/relative).resolve()
        require(not relative.is_absolute() and p.is_relative_to(self.root) and p.is_file(),"Missing or escaped aligned asset: "+str(relative))
        signature=(p.stat().st_size,p.stat().st_mtime_ns,descriptor["sha256"])
        if self._verified.get(p)!=signature:
            require(sha(p)==descriptor["sha256"],"Aligned file SHA256 differs: "+str(relative))
            self._verified[p]=signature
        return p

    def map_descriptor(self,descriptor):
        p=self.checked_path(descriptor)
        a=np.load(p,mmap_mode="r",allow_pickle=False)
        require(isinstance(a,np.memmap) and a.shape==tuple(descriptor["shape"]),"Aligned NPY extent differs")
        if "dtype" in descriptor:require(a.dtype==np.dtype(descriptor["dtype"]),"Aligned NPY dtype differs")
        require(not a.flags.writeable and (a.flags.c_contiguous or a.flags.f_contiguous),"Only read-only contiguous NPY memory maps are accepted")
        return a

    def _validate_axis(self,f):
        axis=f["axis"];kind=axis["axis_kind"]
        require(kind in ("node_group","selected_auxiliary_group","contact_or_component_vector"),"Unsupported signal axis")
        if kind=="contact_or_component_vector":
            require(axis.get("spatial_node_plot_permitted") is False,"Component vector cannot pretend to have cell coordinates")
            return
        n=f["count"]
        require(len(axis["axis_group_rows"])==n and len(f["group_ids"])==n and len(f["centroid_mm"])==n,"Group identity / coordinate columns differ")
        require(np.asarray(f["centroid_mm"]).shape==(n,2) and np.isfinite(f["centroid_mm"]).all(),"Invalid group coordinates")
        require(f["coordinate_role"]==axis["coordinate_role"],"Coordinate roles differ")
        if f["cell_count_not_applicable"]:require(f["members"] is None,"Functional pool cannot claim a cell count")
        else:require(len(f["members"])==n and all(v>0 for v in f["members"]),"Invalid group member counts")

    def open_field(self,phase,fid):
        require(phase in PHASES,"Unknown aligned stream")
        key=(phase,fid)
        if key not in self._mapped:
            f=self.fields[phase][fid]
            if f['axis']['axis_kind']!='contact_or_component_vector':
                pop=f['population'];meta=self.geometry[pop]
                if pop not in self._geometry_loaded:
                    self._geometry_loaded[pop]={name:np.array(self.map_descriptor(meta['arrays'][name]),copy=True)
                                               for name in ('group_ids','counts','centroid_mm')}
                geometry=self._geometry_loaded[pop];rows=f['axis']['axis_group_rows']
                require(np.array_equal(f['group_ids'],geometry['group_ids'][rows]) and
                        np.array_equal(f['centroid_mm'],geometry['centroid_mm'][rows]) and
                        f['coordinate_role']==meta['coordinate_role'],"Selected field / geometry registry differs")
                if not f['cell_count_not_applicable']:
                    require(np.array_equal(f['members'],geometry['counts'][rows]),"Selected field member counts differ")
            a=self.map_descriptor(dict(path=f["values"],sha256=f["values_sha256"],shape=f["shape"],dtype=f["dtype"]))
            require(a.flags.c_contiguous,"Recorded state rows must use the declared C order")
            self._mapped[key]=a
        return self._mapped[key]

    def row_at(self,seconds):
        return int(np.clip(np.searchsorted(self.indices["primary"]["time_source_s"],seconds+1e-12,side="right")-1,0,self.n-1))

    def point(self,phase,row):
        require(0<=row<self.n,"Aligned row out of bounds")
        result={name:self.indices[phase][name][row].item() for name in INDEX_NAMES}
        return {k:v.decode("ascii") if isinstance(v,bytes) else v for k,v in result.items()}

    def sample(self,phase,fid,row):
        a=self.open_field(phase,fid);f=self.fields[phase][fid]
        unit=f["unit"];factor=1000 if unit=="V" else 1e9 if unit=="S" else 1
        values=np.asarray(a[row])*factor
        require(np.isfinite(values).all(),"Selected aligned row contains nonfinite values")
        key=(phase,fid)
        if key not in self._ranges:
            # A bounded shared row subset sets only the display color scale.
            rows=np.linspace(0,self.n-1,min(128,self.n),dtype=int)
            colors=np.asarray(a[rows])*factor
            require(np.isfinite(colors).all(),"Aligned display-range sample contains nonfinite values")
            self._ranges[key]=(float(colors.min()),float(colors.max()))
        return dict(values=values,unit="mV" if unit=="V" else "nS" if unit=="S" else unit,
                    range=self._ranges[key],point=self.point(phase,row),field=f)

    def voltage_field(self,phase,population,group_id):
        for f in self.fields[phase].values():
            if f["kind"]=="ports" and f["key"]==[population,"voltage"] and group_id in f.get("group_ids",[]):
                return f["id"],f["group_ids"].index(group_id)
        return None

    def close(self):
        for a in list(self._mapped.values())+[v for index in self.indices.values() for v in index.values()]:
            a._mmap.close()
        self._mapped.clear();self.indices.clear()

    def release_fields(self,keep):
        for key in list(self._mapped):
            if key not in keep:
                self._mapped.pop(key)._mmap.close()
