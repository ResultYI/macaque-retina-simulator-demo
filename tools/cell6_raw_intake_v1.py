"""Source-bound Cell6 intake; decode only the18 approved SpikeTimes cells."""
from pathlib import Path
from io import BytesIO
import struct
import zlib
import numpy as np
from scipy.io import loadmat


def element(data,pos):
    word=struct.unpack_from('<I',data,pos)[0]
    if word>>16:
        kind,size=word&65535,word>>16
        return kind,data[pos+4:pos+4+size],pos+8
    kind,size=struct.unpack_from('<II',data,pos)
    return kind,data[pos+8:pos+8+size],pos+8+((size+7)//8)*8


def header(matrix):
    kind,_,_=element(matrix,0)
    if kind!=14:raise ValueError('MAT v5 matrix tag required')
    _,flags,p=element(matrix,8);_,dimensions,p=element(matrix,p);_,name,p=element(matrix,p)
    return name.decode('ascii'),np.frombuffer(dimensions,dtype='<i4'),struct.unpack_from('<I',flags)[0]&255,p


def selected_spikes(path,allowed):
    raw=Path(path).read_bytes()
    if raw[126:128]!=b'IM':raise ValueError('Bound little-endian MAT v5 required')
    pos=128
    while pos<len(raw):
        kind,size=struct.unpack_from('<II',raw,pos);payload=raw[pos+8:pos+8+size]
        if kind==15:
            prefix=zlib.decompressobj().decompress(payload,256)
            name,_,_,_=header(prefix)
            if name=='SpikeTimes':matrix=zlib.decompress(payload);break
            pos+=8+size
        elif kind==14:
            matrix=raw[pos:pos+8+size];name,_,_,_=header(matrix)
            if name=='SpikeTimes':break
            pos+=8+((size+7)//8)*8
        else:raise ValueError('Unexpected top-level MAT tag')
    else:raise ValueError('SpikeTimes field absent')
    name,shape,kind,pos=header(matrix)
    if name!='SpikeTimes' or tuple(shape)!=(1,30) or kind!=1:raise ValueError('Expected1x30 MAT cell array')
    result={}
    for row in range(30):
        begin=pos;tag,_,pos=element(matrix,pos)
        if tag!=14:raise ValueError('Expected separate MAT numeric spike cell')
        if row not in allowed:continue
        cell=matrix[begin:pos]
        _,_,q=element(cell,8);_,_,q=element(cell,q)
        name_begin=q;_,_,name_end=element(cell,q)
        name=b'approved_spike';named=struct.pack('<II',1,len(name))+name+b'\0'*((-len(name))%8)
        body=cell[8:name_begin]+named+cell[name_end:]
        decoded=loadmat(BytesIO(raw[:128]+struct.pack('<II',14,len(body))+body))
        values=decoded['approved_spike'].reshape(-1)
        if not np.isfinite(values).all() or not np.array_equal(values,np.round(values)) or np.any(np.diff(values)<=0) or np.any((values<1)|(values>71000)):
            raise ValueError('Original one-based increasing acquisition sample support changed')
        result[row]=values.astype(np.int64)
    if set(result)!=set(allowed):raise ValueError('Selected spike cell intake incomplete')
    return result
