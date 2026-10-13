"""Explicit single-record source operators; no inference of missing physical closure."""
from dataclasses import replace
from pathlib import Path
import json,zipfile
import xml.etree.ElementTree as ET
import numpy as np
from .contracts import ContractError,Quantity
from .mechanism_contracts import EvidenceValue,MechanismProfile,register_mechanism_profile
from .physical import terminal_activation

def maddox_activation_profile(registry,record_id):
    """Select exactly one cell x holding fit and verify both original workbook cells."""
    records=json.loads((Path(registry.root)/'sources/maddox2024_activation_records.json').read_text(encoding='utf-8'))
    selected=[]
    for row in records['records']:
        name='cell%d_hold%d' % (row['source_cell_ordinal'],row['holding_mV'])
        if name==record_id:selected.append(row)
    if len(selected)!=1:raise ContractError('Select an exact Maddox record_id, e.g. cell1_hold-90; no averaging/nearest holding')
    row=selected[0];source=registry.sources['MADDOX2024_fig5_workbook'];source.verify(registry.root)
    with zipfile.ZipFile(Path(registry.root)/source.source_path) as archive:
        xml=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    cells={c.attrib['r']:float(c.find('m:v',ns).text) for c in xml.findall('.//m:c',ns)
        if c.find('m:v',ns) is not None and c.attrib.get('t') not in ('s','str','b')}
    column='K' if row['holding_mV']==-90 else 'L'
    addresses=(column+str(row['source_cell_ordinal']+3),column+str(row['source_cell_ordinal']+10))
    if not np.allclose([cells[a] for a in addresses],[row['Vhalf_mV'],row['k_mV']],rtol=0,atol=1e-12):
        raise ContractError('Maddox paired-fit transcription mismatch')
    sid='MADDOX2024_paired_fit_table'
    if sid not in registry.sources:
        registry.register_source(replace(source,source_id=sid,verification_level='numeric_table_verified',
            source_locator='Sheet1 K/L rows4..6 and11..13 paired original numerical fit cells',
            provenance_notes='Paired original fit coefficients numerically checked against identity-pinned workbook; not primary raw measurements or independent model validation'))
    domain={'cell_types':['cone'],'source_cell_ordinal':row['source_cell_ordinal'],'holding_mV':row['holding_mV'],
        'recording_context':'single recorded cone at selected holding; no region or spectral identity inference',
        'regional_discrepancy':records.get('regional_discrepancy','unknown')}
    entries=[]
    for key,value,meaning in [('Vhalf',row['Vhalf_mV'],'activation_half_voltage'),('slope',row['k_mV'],'activation_voltage_slope')]:
        entries.append(EvidenceValue(key,Quantity([value*.001],'V',meaning,'synthetic_fixture'),
            'published_fit',sid,row['locator'],domain,'Single cell/holding fit; gCa, ECa, kinetics and release calibration absent'))
    profile=MechanismProfile('maddox_activation_'+record_id,'M. mulatta',domain,tuple(entries),(sid,),'published_fit')
    register_mechanism_profile(registry,profile)
    return profile

def maddox_ca_activation(voltage,shift,registry,record_id):
    """Normalized G/Gmax activation only; shift must be supplied explicitly in V."""
    profile_id='maddox_activation_'+record_id
    profile=getattr(registry,'mechanism_profiles',{}).get(profile_id) or maddox_activation_profile(registry,record_id)
    voltage.require('V','physical_membrane_voltage',(1,))
    shift.require('V','activation_threshold_shift',(1,))
    return terminal_activation(voltage,profile.quantity('Vhalf','V','activation_half_voltage',(1,)),
        profile.quantity('slope','V','activation_voltage_slope',(1,)),shift)
