import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_gnn_features import fixture
from gnn_features.build import edge_encoding
from gnn_features.contract import R_FIELDS,encode_numeric
from gnn_features.loader import EdgeStore,CompactEdgeStore,TemporalBatchBuilder,TargetDay,encode_r
from gnn_features.run import feature_type


def setup_builder(tmp_path):
    engine=fixture([('01T00:00',0,0,1),('04T09:00',0,1,10),('04T09:00',0,1,20),
        ('04T09:30',1,2,30),('04T10:00',2,0,40),('04T10:00',1,0,50),('04T10:30',0,2,60)])
    a=engine.arr;folder=tmp_path/'graph/edges';folder.mkdir(parents=True)
    table=pa.table({'edge_id':np.arange(7),**edge_encoding(a,engine.bank)})
    pq.write_table(table,folder/'edges.parquet',row_group_size=2)
    adj={}
    for d,c in [('in','dst'),('out','src')]:
        order=a[d+'_order'];ids=order[a['src'][order]!=a['dst'][order]]
        adj[d+'_edge_ids']=ids;adj[d+'_ptr']=np.r_[0,np.cumsum(np.bincount(a[c][ids],minlength=3))]
    r=[]
    for i in [4,5]:
        values,_=engine.target(i);r.append(dict(**values,full_6h=True,full_day=True))
    schema=pa.schema([(f['name'],feature_type(f['name'])) for f in R_FIELDS]+[('full_6h',pa.bool_()),('full_day',pa.bool_())])
    table=pa.Table.from_pylist(r,schema=schema)
    encoded=encode_r(table)
    for i,row in enumerate(r):np.testing.assert_allclose(encoded[i],encode_numeric(row,R_FIELDS)+[1,1],rtol=1e-6)
    return engine,TemporalBatchBuilder(engine,adj,EdgeStore(tmp_path,max_groups=2),np.arange(3)),encoded


def test_temporal_batch_reads_saved_features_and_cache(tmp_path,monkeypatch):
    engine,builder,r=setup_builder(tmp_path)
    monkeypatch.setattr(engine,'target',lambda *args:pytest.fail('R must not be recomputed'))
    first,y,cold=builder.build([4,5],r,[0,1])
    second,_,warm=builder.build([4,5],r,[1,0])
    assert set(first['edge_id'])=={1,2,3} # Keep parallel edges; exclude same-time peers, future, self.
    assert warm['node_cache_misses']==0 and cold['node_cache_misses']==3
    assert not any('label'==k or k=='y' for k in first)
    for name in first:np.testing.assert_array_equal(first[name],second[name])
    np.testing.assert_array_equal(y,[0,1])
    with pytest.raises(ValueError,match='Mixed cutoff'):builder.build([3,4],r,[0,1])
    # Same node at a later cutoff must not reuse the earlier feature vector.
    _,_,later=builder.build([6],r[:1],[0])
    assert later['node_cache_misses']>0


def test_empty_history_and_exact_cutoff_iterator(tmp_path):
    engine,builder,r=setup_builder(tmp_path)
    tensors,_,_=builder.build([0],r[:1],[0])
    assert tensors['edge_index_in'].shape==(2,0)
    assert tensors['edge_attr_num'].shape==(0,9)
    day=TargetDay.__new__(TargetDay);day.ids=np.array([7,8,9,10]);day.times=np.array([20,10,20,10]);day.order=np.lexsort((day.ids,day.times))
    batches=list(day.batches(1))
    assert [int(b[0]) for b in batches]==[8,10,7,9]
    assert sum(map(len,day.batches(3)))==4
    with pytest.raises(ValueError):day.positions([11])


def test_compact_cache_equals_persisted_edges_and_rejects_stale_archive(tmp_path):
    import json
    _,builder,_=setup_builder(tmp_path)
    (tmp_path/'build.json').write_text(json.dumps({'rows':7}),encoding='utf-8')
    cache=tmp_path/'cache';compact=CompactEdgeStore(tmp_path,cache)
    for x,y in zip(builder.edges.get([6,1,4,1]),compact.get([6,1,4,1])):np.testing.assert_array_equal(x,y)
    assert compact.get([])[0].shape==(0,8)
    CompactEdgeStore(tmp_path,cache)
    (tmp_path/'build.json').write_text(json.dumps({'rows':8}),encoding='utf-8')
    with pytest.raises(ValueError,match='Incompatible'):CompactEdgeStore(tmp_path,cache)


def test_prepared_graph_and_node_lookup_avoid_recomputation(tmp_path,monkeypatch):
    engine,builder,r=setup_builder(tmp_path)
    original,y,_=builder.build([4,5],r,[0,1])
    cached=dict(builder.node_cache);graph=(original['node_id'],original['edge_id'])
    builder.clear_cache()
    monkeypatch.setattr(builder,'sample',lambda *args:pytest.fail('Prepared graph must not resample'))
    monkeypatch.setattr(engine,'node',lambda *args:pytest.fail('Prepared N must not query history'))
    actual,z,_=builder.build([4,5],r,[0,1],prepared=graph,node_lookup=lambda node,t:cached[(t,node)])
    for key in original:np.testing.assert_array_equal(original[key],actual[key])
    np.testing.assert_array_equal(y,z)
