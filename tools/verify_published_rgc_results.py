"""Verify the selected repository evidence, without private baseline or simulation."""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import whosmat

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/modeling_scope_20261020'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'docs/rgc_oct20_publication_manifest.json').read_bytes())
    for f in manifest['files']:
        p=ROOT/f['path'];assert p.is_file(),f['path']
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
    source=ROOT/'data/source_data/latimer2019_figshare_v1'
    tree=json.loads((source/'author_repository_tree.json').read_bytes());blobs={x['path']:x['sha'] for x in tree['tree']}
    for local,remote in [('author_pyCBEM__RGC_CBEM.py','pyCBEM/RGC_CBEM.py'),('author_pyCBEM__utils.py','pyCBEM/utils.py'),('LICENSE.CBEM-MIT.txt','LICENSE')]:
        raw=(source/local).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blobs[remote],local
    assert {name for name,shape,kind in whosmat(source/'Cell_6.mat')}=={'ExcCurrents','ExcStm','InhCurrents','InhStm','SpikeStm','SpikeTimes','fn'}
    assert sha(source/'Cell_6.mat')=='59e1457bc27f7392a3c458902941a234c687b26a525b26d168bc8609aae40025'
    audit=json.loads((BASE/'rgc_voltage_hazard_source_audit_v1/result.json').read_bytes())
    table=list(csv.DictReader((BASE/'rgc_voltage_hazard_source_audit_v1/eight_state_stationary.csv').open(encoding='utf-8')))
    assert abs(sum(float(x['stationary_probability']) for x in table)-1)<1e-13
    expected=sum(float(x['weighted_count_rate_Hz']) for x in table)
    assert abs(expected-135.38217292416522)<1e-11
    for case in audit['cases'].values():assert abs(case['stationary_count_rate_Hz']-expected)<1e-11
    ref=BASE/'on_parasol_author_reference_v1'
    matrix=np.loadtxt(ref/'zero_history_mapping.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(matrix[:,1],90*np.logaddexp(0.,(matrix[:,0]+53)/1.67),rtol=1e-14,atol=1e-12)
    case=BASE/'on_parasol_candidate_network_v1'
    params=json.loads((case/'actual_RGC_parameter_snapshots.json').read_bytes())
    delta={cid:[k for k in params['old'][cid] if params['old'][cid][k]!=params['candidate'][cid][k]] for cid in params['old']}
    assert delta['parasol_RGC_ON']==['alpha','mu','beta']
    assert all(not fields for cid,fields in delta.items() if cid!='parasol_RGC_ON')
    actual=params['candidate']['parasol_RGC_ON']
    assert actual['alpha']==[90.]*3 and actual['mu']==[-.053]*3 and actual['beta']==[.00167]*3
    with np.load(case/'old/stimulus/traces.npz',allow_pickle=False) as a,np.load(case/'candidate/stimulus/traces.npz',allow_pickle=False) as b:
        for k in a.files:
            if k.startswith('parasol_RGC_ON.') and not k.endswith(('.voltage','.gE','.gI')):continue
            assert np.array_equal(a[k],b[k]),k
    for variant in ['old','candidate']:
        with np.load(case/variant/'stimulus/traces.npz',allow_pickle=False) as stim,np.load(case/variant/'matched_gray/traces.npz',allow_pickle=False) as gray,\
             np.load(case/variant/'exact_expected_count_rates.npz',allow_pickle=False) as sr,np.load(case/variant/'matched_gray/exact_expected_count_rates.npz',allow_pickle=False) as gr,\
             np.load(case/variant/'matched_gray/matched_response_deltas.npz',allow_pickle=False) as diff:
            for cid in params[variant]:
                np.testing.assert_array_equal(diff[cid+'.voltage_delta_mV'],1000*(stim[cid+'.voltage']-gray[cid+'.voltage']))
                np.testing.assert_array_equal(diff[cid+'.expected_rate_delta_Hz'],sr[cid]-gr[cid])
    regression=ET.parse(case/'candidate_regression.xml').getroot()[0].attrib
    reference_tests=ET.parse(ref/'reference_checks.xml').getroot()[0].attrib
    assert regression['tests']=='11' and reference_tests['tests']=='4'
    assert all(t['errors']==t['failures']=='0' for t in (regression,reference_tests))
    external=[dict(path=x['path'],present=(ROOT/x['path']).is_file()) for x in manifest['external_assets']]
    print(json.dumps(dict(status='PUBLISHED_EVIDENCE_VERIFIED',files=len(manifest['files']),
        bytes=sum(x['bytes'] for x in manifest['files']),source_code_blobs_and_license_verified=3,
        public_Cell6_schema_verified=True,source_mapping_verified=True,actual_parameter_delta=delta,
        saved_network_arrays_and_matched_subtraction_verified=True,prior_regression_tests=15,
        external_assets=external,new_simulation_seconds=0,matlab_launches=0,physiological_calibration=False),indent=2))


if __name__=='__main__':main()
