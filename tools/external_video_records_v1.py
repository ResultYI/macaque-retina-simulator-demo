"""Read original port arrays across lossless, nonoverlapping saved time chunks."""
from contextlib import ExitStack,contextmanager
import hashlib,json
from pathlib import Path
import numpy as np


class RecordedFrames:
    def __init__(self,parts):
        self.parts=parts;self.files=parts[0].files
        if any(p.files!=self.files for p in parts):raise ValueError('Time chunks have different port registries')
    def __getitem__(self,key):
        return np.concatenate([part[key] for part in self.parts],axis=0 if key in ('time_seconds','side') else 1)


@contextmanager
def open_response(folder: Path):
    with ExitStack() as stack:
        if (folder/'RESPONSE.npz').is_file():
            yield stack.enter_context(np.load(folder/'RESPONSE.npz',allow_pickle=False));return
        index=json.loads((folder/'RESPONSE_INDEX.json').read_bytes());parts=[]
        for entry in index['chunks']:
            path=(folder/entry['file']).resolve()
            if not path.is_relative_to(folder.resolve()):raise ValueError('Chunk outside declared run')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Saved time chunk differs')
            parts.append(stack.enter_context(np.load(path,allow_pickle=False)))
        frames=RecordedFrames(parts)
        if len(frames['time_seconds'])!=index['records']:raise ValueError('Saved record count differs')
        yield frames
