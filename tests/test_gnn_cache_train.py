import json
import numpy as np
import pytest
from gnn_features.contract import HOUR
from gnn_features.loader import TargetDay
from gnn_features.cache_train import chunks, validate_existing, array_sha, sha, clear_work


def test_blocks_preserve_timestamp_batches_and_hour_boundaries():
    day=TargetDay.__new__(TargetDay)
    day.ids=np.arange(150)
    day.times=np.r_[np.full(130,HOUR-1),np.full(20,HOUR)]
    day.order=np.lexsort((day.ids,day.times))
    blocks=list(chunks(day,limit=2))
    assert [(h,i,len(b)) for h,i,b in blocks]==[(0,0,2),(0,1,1),(1,0,1)]
    np.testing.assert_array_equal(np.concatenate([x for _,_,bs in blocks for x in bs]),day.ids)
    for hour,_,batches in blocks:
        for ids in batches:
            assert len(ids)<=64 and len(np.unique(day.times[ids]))==1
            assert np.all(day.times[ids]//HOUR==hour)


def test_resume_rejects_changed_scope_and_corrupt_outputs(tmp_path):
    ids=np.array([1,2]);payload=tmp_path/'nodes.parquet';payload.write_bytes(b'checksum fixture')
    manifest=dict(status='verified',config_sha256='config',target_ids_sha256=array_sha(ids),targets=2,file_sha256={'nodes.parquet':sha(payload)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    assert validate_existing(tmp_path,'config',ids)==manifest
    with pytest.raises(ValueError,match='identity/config'):validate_existing(tmp_path,'different',ids)
    with pytest.raises(ValueError,match='identity/config'):validate_existing(tmp_path,'config',ids[::-1])
    payload.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):validate_existing(tmp_path,'config',ids)


def test_scratch_cleanup_leaves_published_data(tmp_path):
    work=tmp_path/'_work'/'active';work.mkdir(parents=True);(work/'keys.parquet').write_bytes(b'partial')
    published=tmp_path/'transaction_date=2022-09-01';published.mkdir();(published/'nodes.parquet').write_bytes(b'verified')
    clear_work(tmp_path)
    assert not work.exists()
    assert (published/'nodes.parquet').read_bytes()==b'verified'
