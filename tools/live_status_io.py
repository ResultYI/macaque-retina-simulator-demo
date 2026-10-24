"""One normal retry for a transient owned live-status replacement collision.

Immutable research artifacts still use the original no-replacement writer.
No permission, ACL or resource setting is changed.
"""
from pathlib import Path
import json
import os
import time

from retina_sim.split_chunk_store import write_json as _atomic_write


def read_json(path):
    with Path(path).open('rb') as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2:raise ValueError('Live status exceeds its8MiB bound')
    return json.loads(raw)


def write_json(path,value,*,replace=False):
    path=Path(path)
    try:return _atomic_write(path,value,replace=replace)
    except PermissionError as exc:
        # Only retry an already-fsynced owned temporary file replacing this
        # same live status. An open/create/ACL denial is never reinterpreted.
        if (not replace or os.name!='nt' or exc.winerror not in (5,32)
                or not exc.filename or not exc.filename2):raise
        temporary=Path(exc.filename).resolve();target=Path(exc.filename2).resolve()
        if (target!=path.resolve() or temporary.parent!=target.parent
                or not temporary.name.startswith(target.name+'.writing-')
                or not temporary.is_file()):raise
        time.sleep(.025)
        os.replace(temporary,target)  # One normal retry; a second denial propagates.
