"""Fixed ISET lattice assets with explicit species and RF/anatomy roles.

The upstream algorithm is a numerical mosaic generator, not a macaque density
source. Borrowing its official human lattice requires a separate named schema;
it is never an implicit replacement for the supplied macaque density.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .contracts import ContractError,Quantity,canonical_hash

COMMIT='d2924c2fe1d41c7dc5af1006d8081735b08f1b35'
SCHEMA='isetbio_macaque_lattice_master_v1'
HUMAN_SCHEMA='isetbio_official_human_borrowed_master_v1'
HUMAN_ASSET_SHA256='54eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728'
HUMAN_TRANSFORM={'negate_raw_coordinates':True,'microns_to_mm':.001,'resampling':False,'spatial_warp':False}
MIDGET_CENTROID_SOURCE={'repository':'https://github.com/isetbio/isetbio','commit':COMMIT,
    'file':'ganglioncells/@MosaicConnector/updateDestinationCentroidsFromInputs.m',
    'locator':'connectedSourceRFIndices/weightedMean, lines12-33 and39-52',
    'evidence':'published_code_geometry_algorithm; not macaque midget density or soma data'}

def _require(ok,message):
    if not ok:raise ContractError('ISET_GEOMETRY: '+message)

def connected_cone_rf_centers(cone_positions,weights,*,orientation='post_by_cone'):
    """ISET-style connected-input centroid; returns functional centers, not soma.

    ISET's cone-by-RGC connectivity matrix is accepted only with explicit axis
    selection. Project contact matrices use postsynaptic rows by cone columns.
    No population density, synapse or anatomical displacement is inferred here.
    """
    cone_positions.require('mm','functional_position',frame='functional_visual_field')
    xy=cone_positions.values;_require(xy.ndim==2 and xy.shape[1]==2,'Finite Nx2 cone RF positions required')
    _require(orientation in ('post_by_cone','iset_cone_by_post'),'Explicit connectivity orientation required')
    w=sparse.csr_matrix(weights,dtype=np.float64,copy=True)
    if orientation=='iset_cone_by_post':w=w.T.tocsr()
    w.eliminate_zeros()
    _require(w.shape[1]==len(xy) and np.all(np.isfinite(w.data)) and np.all(w.data>=0),'Nonnegative finite cone weights/layout mismatch')
    sums=np.asarray(w.sum(axis=1)).ravel()
    _require(np.all(sums>0),'Unresolved RF center: an actual connected cone is required for every cell')
    result=np.asarray(w@xy)/sums[:,None]
    # A private line retains its exact input location, including signed zeros.
    single=np.diff(w.indptr)==1
    if np.any(single):result[single]=xy[w.indices[w.indptr[:-1][single]]]
    return Quantity(result,'mm','functional_position','connected_cone_functional_centroid_project_adapter','functional_visual_field')

def cone_apertures(positions_mm,*,support_ids=None,selected_ids=None,spacing_factor=.79):
    """Same explicit project aperture convention used by the active ISET front end.

    Recomputed on the new complete support; never transfer old apertures by row.
    """
    xy=np.asarray(positions_mm,dtype=float)
    _require(xy.ndim==2 and xy.shape[1]==2 and len(xy)>=7 and np.all(np.isfinite(xy)),'At least7 finite full-support cone positions required')
    _require(len(np.unique(xy,axis=0))==len(xy),'Duplicate cone positions')
    d,_=cKDTree(xy).query(xy,k=7);diameters=spacing_factor*np.median(d[:,1:],axis=1)*1000
    if selected_ids is not None:
        _require(support_ids is not None and len(support_ids)==len(xy) and len(set(support_ids))==len(xy),'Unique full-support IDs required')
        lookup={key:i for i,key in enumerate(support_ids)}
        _require(all(key in lookup for key in selected_ids),'Selected cone absent from complete aperture support')
        diameters=diameters[[lookup[key] for key in selected_ids]]
    return diameters

def input_geometry_support(root,record):
    """Return the declared complete aperture support, keyed by actual cone ID."""
    if 'cone_master' not in record['atlas']:
        support=record['cones']
        return list(support['global_cell_ids']),np.asarray(support['coordinates']['rf']['values'],dtype=float)
    from .source_atlas import SourceDensityAtlas
    atlas=SourceDensityAtlas(root,profile_id=record.get('isetbio_geometry',{}).get('atlas_profile','retina_atlas_isetbio_macaque_v1'))
    _require(dict(atlas.config['cone_master'])==record['atlas']['cone_master'],'Input support master differs from neural geometry')
    _require(atlas.mosaic_namespace==record['atlas']['mosaic_namespace'],'Input support namespace differs from neural geometry')
    master=atlas.cone_master;table=atlas._table('cones',master.source_keys,master.positions)
    return list(table.global_cell_ids),master.positions

class FixedISETConeMaster:
    def __init__(self,root,binding):
        root=Path(root);path=root/binding['path'];raw=path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Frozen master manifest hash mismatch')
        self.manifest=json.loads(raw);m=self.manifest
        _require(m.get('schema') in (SCHEMA,HUMAN_SCHEMA) and m.get('isetbio_commit')==COMMIT,'Unsupported pinned generator/schema')
        if m['schema']==SCHEMA:
            _require(m.get('human_density_used') is False and m.get('density_species')=='Macaca nemestrina','Explicit macaque cone density required')
        else:
            _require(m.get('human_density_used') is True and m.get('density_species')=='Homo sapiens'
                and m.get('geometry_route')=='official_human_borrowed_approximation_v1','Explicit human-borrowed approximation required')
            source=m.get('official_asset',{})
            _require(source.get('sha256')==HUMAN_ASSET_SHA256 and source.get('eye')=='right eye'
                and source.get('nominal_FOV_degrees')==58,'Only the explicitly authorized official human asset is supported')
            _require(m.get('coordinate_transform')==HUMAN_TRANSFORM,'Human coordinates must retain the official sign/unit transform without warp/resampling')
        _require(m.get('population')=='cones' and m.get('position_role')=='rf_input_inner_segment_project_transfer','Functional input positions are not soma')
        payload=root/m['payload']['path'];data=payload.read_bytes()
        _require(hashlib.sha256(data).hexdigest()==m['payload']['sha256'],'Frozen lattice payload hash mismatch')
        with np.load(payload,allow_pickle=False) as a:
            self.positions=np.asarray(a['positions_mm'],dtype=np.float64).copy()
            self.source_keys=tuple(str(v) for v in a['source_keys'])
        _require(self.positions.shape==(len(self.source_keys),2) and np.all(np.isfinite(self.positions)),'Invalid lattice layout')
        _require(len(set(self.source_keys))==len(self.source_keys),'Duplicate master source identity')
        _require(len(np.unique(self.positions,axis=0))==len(self.positions),'Duplicate lattice positions')
        self.positions.setflags(write=False);self.bounds=tuple(m['complete_query_bounds_mm'])
        self.identity=binding['sha256'];self._tiles={};self._tile_indices={}
    def tile(self,tx,ty,size):
        x0,x1,y0,y1=tx*size,(tx+1)*size,ty*size,(ty+1)*size;b=self.bounds
        _require(x0>=b[0]-1e-12 and x1<=b[1]+1e-12 and y0>=b[2]-1e-12 and y1<=b[3]+1e-12,
            'Query exceeds fixed master support; generate an explicitly larger shared master, never regenerate per ROI')
        key=(tx,ty,size)
        if key not in self._tiles:
            if size not in self._tile_indices:
                ij=np.floor(self.positions/size).astype(np.int64)
                order=np.lexsort((np.arange(len(ij)),ij[:,1],ij[:,0]));sorted_ij=ij[order]
                edges=np.r_[0,np.flatnonzero(np.any(np.diff(sorted_ij,axis=0),axis=1))+1,len(order)]
                self._tile_indices[size]={(int(sorted_ij[a,0]),int(sorted_ij[a,1])):order[a:b]
                    for a,b in zip(edges[:-1],edges[1:])}
            buckets=self._tile_indices[size]
            candidates=[buckets[(tx+dx,ty+dy)] for dx in (-1,0,1) for dy in (-1,0,1) if (tx+dx,ty+dy) in buckets]
            selected=np.sort(np.concatenate(candidates)) if candidates else np.empty(0,dtype=np.int64)
            # Retain the exact original comparisons at floating tile boundaries.
            xy=self.positions[selected];keep=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
            selected=selected[keep]
            self._tiles[key]=(tuple(self.source_keys[i] for i in selected),self.positions[selected])
        return self._tiles[key]
    def require_bounds(self,bounds):
        x0,x1,y0,y1=bounds;b=self.bounds
        _require(b[0]<=x0<x1<=b[1] and b[2]<=y0<y1<=b[3],'ROI/halo is outside this shared geometry master')
