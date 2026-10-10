"""Stored motif counts must remain bound to transaction identity and encoding."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gnn_features import v4_features as f


@pytest.fixture
def block(tmp_path):
    ids=np.array([5,2,8],np.int64);times=np.array([100,101,102],np.int64)
    raw=np.array([[0,4,9],[-1,-1,0],[-1,-1,-1]],np.int64)
    table=pa.table({'edge_id':ids,'cutoff_us':times,**{n:raw[:,i] for i,n in enumerate(f.NAMES)}})
    pq.write_table(table,tmp_path/'features.parquet')
    m=dict(status='verified',config_sha256='cfg',base_manifest_sha256='base',
        contract_sha256=f.digest(f.contract()),targets=3,target_ids_sha256=f.array_sha(ids),
        file_sha256=f.file_sha(tmp_path/'features.parquet'))
    f.atomic_json(tmp_path/'manifest.json',m)
    return tmp_path,ids,times,raw,m


def test_integer_counts_roundtrip_exact_and_encoding(block):
    path,ids,times,raw,m=block
    got,encoded,support=f._validate_sidecar(path,'cfg','base',ids,times)
    assert got==m
    np.testing.assert_array_equal(encoded,f.encode_extra(raw))
    np.testing.assert_array_equal(np.column_stack([support[n] for n in f.NAMES]),raw)


@pytest.mark.parametrize('change',['id','time','config','base','contract','file','count'])
def test_corruption_never_silently_reused(block,change):
    path,ids,times,raw,m=block
    if change=='id':ids=ids[::-1]
    if change=='time':times=times+1
    if change=='config':m['config_sha256']='other'
    if change=='base':m['base_manifest_sha256']='other'
    if change=='contract':m['contract_sha256']='other'
    if change=='file':m['file_sha256']='other'
    if change=='count':m['targets']=2
    f.atomic_json(path/'manifest.json',m)
    with pytest.raises(ValueError):f._validate_sidecar(path,'cfg','base',ids,times)


def test_sidecar_reordered_lookup_and_wrong_cutoff(block):
    path,ids,times,raw,m=block
    store=object.__new__(f.SidecarStore)
    order=np.argsort(ids)
    store.blocks={'b':{}};store._loaded='b';store._ids=ids[order];store._times=times[order]
    store._matrix=f.encode_extra(raw)[order];store._support={n:raw[:,j][order] for j,n in enumerate(f.NAMES)}
    np.testing.assert_array_equal(store.block('b',ids[::-1],times[::-1]),f.encode_extra(raw)[::-1])
    with pytest.raises(ValueError):store.block('b',ids,times+1)
    with pytest.raises(ValueError):store.block('b',np.array([999]),np.array([0]))
    with pytest.raises(ValueError):store.block('b',np.array([2,2]),np.array([101,101]))
