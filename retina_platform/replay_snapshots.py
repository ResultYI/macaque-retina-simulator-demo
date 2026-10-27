"""Selected fields from real saved checkpoints, without restoring a simulator."""
import hashlib
import json

import numpy as np

from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import load_bundle
from .replay_data import read_json, require, sha


def mapping(encoded):
    def key(value):
        return tuple(value["tuple"]) if isinstance(value,dict) and "tuple" in value else value
    return {key(k):v for k,v in encoded["dictionary"]}


class SparseSnapshots:
    def __init__(self, data):
        prepared=data.root/data.spec["prepared_model"]
        seed_path=prepared/"initial_seed"
        require(sha(seed_path/"manifest.json")==data.spec["prepared_manifest_sha256"],"Frozen seed manifest differs")
        seed_bundle=load_bundle(seed_path,expected_kind="source_construction_seed")
        seed=seed_bundle["payload"]
        require(canonical_hash(seed)==data.spec["source_seed_sha256"],"Frozen seed payload differs")
        self.origin=seed["time_seconds"]
        static=data.root/"reports/ac_vs_native_support_v1/static_candidate_v1"
        descriptor=read_json(static/"selection_identity.json")
        require(descriptor["configuration"]==seed_bundle["identity"]["candidate_config"],"Static selection configuration differs")
        with np.load(static/"selection.npz",allow_pickle=False) as saved:
            selection={k:saved[k].copy() for k in saved.files}
        require(canonical_hash(selection)==descriptor["selection_sha256"],"Static selection rows differ")
        originals=[]
        for name in ("inner_selection","native_selection"):
            binding=descriptor["configuration"][name];path=data.root/binding["path"]
            require(sha(path)==binding["sha256"],"Frozen original selection differs")
            with np.load(path,allow_pickle=False) as saved:originals.append({k:saved[k].copy() for k in saved.files})
        expected=dict(originals[0])
        for pop in ("cones","HC"):expected["population__"+pop]=originals[1]["population__"+pop]
        require(set(expected)==set(selection) and all(np.array_equal(expected[k],v) for k,v in selection.items()),"Candidate rows differ from frozen native060 / inner030 selection")
        base_config=read_json(data.root/descriptor["configuration"]["base_patch_config"])
        parent_path=data.root/base_config["source_artifact"]["path"]
        require(sha(parent_path)==base_config["source_artifact"]["sha256"],"Parent geometry manifest differs")
        parent=read_json(parent_path)
        require(parent["manifest_sha256"]==descriptor["parent_artifact"],"Parent geometry identity differs")
        self.geometry={}
        def parent_array(pop,field):
            rel=f"populations/{pop}/{field}.npy"
            p=parent_path.parent/rel
            require(sha(p)==parent["files"][rel]["sha256"],"Group geometry array differs: "+rel)
            return np.load(p,allow_pickle=False,mmap_mode="r")[selection["population__"+pop]].copy()
        used={k[0] for k in seed["states"] if k[0] not in data.spec["recording"]}
        for pop in sorted(used & set(parent["populations"])):
            meta=parent["populations"][pop]
            self.geometry[pop]=dict(xy=parent_array(pop,"centroid_mm"),ids=parent_array(pop,"group_ids"),
                members=parent_array(pop,"counts"),coordinate=meta["coordinate_role"],effective=False)
        config=read_json(data.root/"configurations/ac_vs_functional_v1.json")
        for channel in config["channels"]:
            target=data.spec["recording"][channel["target"]]
            pop=channel["pool"]
            self.geometry[pop]=dict(xy=np.asarray(target["centroid_mm"]),ids=np.asarray([pop+":"+i for i in target["global_ids"]]),
                members=np.ones(len(target["rows"]),dtype=int),coordinate="effective_functional_RF_pool_not_anatomical_AC",effective=True)
        monitor=read_json(data.cache_path/"manifest.json")["raw_absolute_monitor"]["states"]
        units={row["state"]:row["unit"] for row in monitor}
        self.fields={};self.values={};self.times={}
        native_mechanism="reduced_native"
        for key,value in seed["states"].items():
            pop,mechanism,field=key
            if pop not in self.geometry or np.asarray(value).shape!=(len(self.geometry[pop]["ids"]),): continue
            if pop in ("cones","HC") and mechanism!=native_mechanism:continue
            if pop not in ("cones","HC") and mechanism not in ("rod_membrane","Chen2024PrimateRodCurrent","ReceptorConductanceMembrane","HistoryBipolarTerminal","DynamicAmacrineInhibition","FunctionalAmacrineVoltageGate_v1"): continue
            unit=units.get(":".join(key))
            if unit is None:continue
            label=f"{field} / "+("mV" if unit=="V" else "nS" if unit=="S" else unit)
            self.fields.setdefault(pop,{})[label]=dict(key=key,unit=unit)
            self.values[key]=[np.asarray(value).copy()];self.times[key]=[0.]
        self.geometry={k:v for k,v in self.geometry.items() if k in self.fields}
        paths=list(data.cache_path.glob("checkpoint_*/manifest.json"))
        # g2 reused its original 50 ms checkpoint; verify the identical cache identity.
        if data.case_id=="cell6_g2":
            extra=data.source/"cache/cell6_g2/checkpoint_0000500/manifest.json"
            if extra.exists() and extra.parent.name not in {p.parent.name for p in paths}:paths.append(extra)
        self.manifests=[]
        for path in sorted(paths,key=lambda p:p.parent.name):
            m=read_json(path)
            require(m.get("schema")=="retipath-group-time-binary-bundle-1" and m.get("kind")=="complete_model_checkpoint" and
                    m["identity"]==dict(cache_identity=data.caches[data.case_id]["cache_identity"]),"Checkpoint identity differs: "+str(path))
            # Encoded manifests contain JSON primitives only. The C JSON encoder
            # preserves canonical bytes without a slow per-scalar Python walk.
            encoded=json.dumps({k:v for k,v in m.items() if k!="manifest_hash"},sort_keys=True,allow_nan=False,separators=(",",":"))
            require(hashlib.sha256(encoded.encode()).hexdigest()==m["manifest_hash"],"Checkpoint manifest hash differs")
            del encoded
            payload=mapping(m["payload"])
            native=mapping(payload["native"]);point=mapping(native["point"])
            clock=mapping(payload["clocks"])
            inner_time=mapping(clock["last_commit_times"])["rod"]-self.origin
            native_time=point["time_seconds"]-self.origin
            inner=mapping(payload["inner"])
            loaded={}
            def array(ref):
                name=ref["array"]
                if name not in loaded:
                    p=path.parent/name;require(sha(p)==m["files"][name]["sha256"],"Checkpoint state bytes differ: "+str(p))
                    loaded[name]=np.load(p,allow_pickle=False)
                return loaded[name]
            # Native accepted point may be a fraction of a millisecond ahead of
            # inner time. Preserve its own timestamp; never label it simultaneous.
            interval=mapping(native["intervals"][-1])
            names,sizes=interval["state_names"],interval["state_sizes"]
            offsets=np.r_[0,np.cumsum(sizes)]
            y=array(point["y"])
            for key in self.values:
                pop,mechanism,field=key
                if pop in ("cones","HC"):
                    require(field in names,"Native state layout missing: "+field)
                    i=names.index(field);value=y[offsets[i]:offsets[i+1]];stamp=native_time
                else:
                    require(key in inner,"Saved inner state missing: "+str(key))
                    value=array(inner[key]);stamp=inner_time
                require(value.shape==self.values[key][0].shape and np.isfinite(value).all(),"Checkpoint state layout differs")
                require(stamp>self.times[key][-1],"Checkpoint state timestamps are not strictly increasing")
                self.values[key].append(value.copy());self.times[key].append(stamp)
            self.manifests.append(dict(path=str(path),manifest_hash=m["manifest_hash"],inner_source_s=inner_time,native_source_s=native_time))
        for key in self.values:
            self.values[key]=np.asarray(self.values[key]);self.times[key]=np.asarray(self.times[key])

    def sample(self, population, label, seconds):
        meta=self.fields[population][label];key=meta["key"]
        times=self.times[key]
        i=int(np.clip(np.searchsorted(times,seconds+1e-12,side="right")-1,0,len(times)-1))
        unit=meta["unit"]
        factor=1000 if unit=="V" else 1e9 if unit=="S" else 1
        display_unit="mV" if unit=="V" else "nS" if unit=="S" else unit
        values=self.values[key]*factor
        return dict(values=values[i],unit=display_unit,stamp=float(times[i]),age=max(0.,seconds-times[i]),
                    range=(float(values.min()),float(values.max())),key=key)
