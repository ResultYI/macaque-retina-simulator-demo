"""Official two-column VR ASCII spike import; no simulator or rate calibration.

The fixed source recorder emits MVASpike integer ID and time in seconds.  The
caller must supply the ID order verified against the saved retina.xml.  Files
are separate explicit repeats; the format itself does not contain trial IDs.
"""
from pathlib import Path
import hashlib
import re
import numpy as np
from .contracts import ContractError


def repeated_ascii_counts(paths, unit_ids, bin_edges_seconds, *, stimulus_onset_seconds=0.):
    paths=[Path(p) for p in paths];ids=tuple(unit_ids);edges=np.asarray(bin_edges_seconds,dtype=float)
    if len(paths)<2 or len({p.resolve() for p in paths})!=len(paths):
        raise ContractError('Separate explicit repeat files required; one train is not a PSTH')
    if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):
        raise ContractError('Explicit unique integer unit IDs from actual retina.xml required')
    if edges.ndim!=1 or len(edges)<2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0):
        raise ContractError('Finite increasing bin edges in seconds required')
    if not np.isfinite(stimulus_onset_seconds):raise ContractError('Recorded stimulus onset in seconds required')
    raw_edges=edges+stimulus_onset_seconds;index={v:i for i,v in enumerate(ids)}
    counts=np.zeros((len(paths),len(edges)-1,len(ids)),dtype=np.int64);reports=[]
    for repeat,path in enumerate(paths):
        raw=path.read_bytes();ignored=0;outside=0;events=0
        for line_no,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
            if not line.strip():continue
            columns=line.split()
            if len(columns)!=2 or re.fullmatch(r'[0-9]+',columns[0]) is None:
                raise ContractError(f'Expected unit_ID and time_seconds at {path.name}:{line_no}')
            try:t=float(columns[1])
            except ValueError as exc:raise ContractError('Invalid spike time') from exc
            if not np.isfinite(t) or t<0:raise ContractError('Spike time must be finite nonnegative seconds')
            unit=int(columns[0]);events+=1
            if unit not in index:ignored+=1;continue
            if t<raw_edges[0] or t>=raw_edges[-1]:outside+=1;continue
            k=int(np.searchsorted(raw_edges,t,side='right')-1)
            counts[repeat,k,index[unit]]+=1
        reports.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),events=events,
            unmapped_unit_events=ignored,mapped_events_outside_fixed_window=outside))
    return counts,dict(source_format='MVASpike ID time_seconds',time_unit='s',unit_ids=list(ids),
        binning='left_closed_right_open',stimulus_onset_seconds=float(stimulus_onset_seconds),
        selection='explicit ID subset; no guessed channel order',
        unit_mapping_validation='Caller must bind these actual unit IDs and positions to the saved retina.xml identity',
        source_files=reports,rate_conversion='Mean repeats divided by declared bin width; no analog conversion')
