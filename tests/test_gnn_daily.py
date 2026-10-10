import pickle
import sys
import pytest
import pyarrow as pa
from types import SimpleNamespace
import pyarrow.parquet as pq
from test_gnn_features import fixture
from gnn_features.partitioned import target_side
from gnn_features.daily import merge_partition,merge_parquet_file


@pytest.mark.parametrize('file_transport',[False,True])
def test_streaming_merge_preserves_exact_features_and_self_nulls(tmp_path,monkeypatch,file_transport):
    context=SimpleNamespace(taskAttemptId=lambda:7)
    monkeypatch.setitem(sys.modules,'pyspark',SimpleNamespace(TaskContext=SimpleNamespace(get=lambda:context)))
    e=fixture([('04T08:00',0,1,8_000_000_000_000_000_000),('04T08:00',0,1,8_000_000_000_000_000),('04T09:00',1,2,99),('04T09:00',1,1,99)])
    expected={};rows=[]
    for i in [2,3]:
        expected[i]=e.target(i)[0]
        for side,node in [('src',int(e.arr['src'][i])),('dst',int(e.arr['dst'][i]))]:
            n=e.node(node);r,audit=target_side(e,i,side,n)
            rows.append(SimpleNamespace(edge_id=i,side=side,cutoff=int(e.arr['timestamp'][i]),transaction_id=f'tx-{i}',payload=pickle.dumps((n,r,audit))))
    (tmp_path/'features').mkdir();(tmp_path/'audit').mkdir()
    if file_transport:
        path=tmp_path/'sorted_input.parquet'
        pq.write_table(pa.Table.from_pylist([vars(row) for row in rows]),path,row_group_size=1)
        result=list(merge_parquet_file(0,path,str(tmp_path/'features'),e.data_start))[0]
    else:
        result=list(merge_partition(0,iter(rows),str(tmp_path/'features'),e.data_start))[0]
    assert result['rows']==2
    table=pq.ParquetFile(result['path']).read().to_pylist()
    for row in table:
        for name,value in expected[row['edge_id']].items():assert row[name]==value
    assert pq.ParquetFile(result['audit_path']).metadata.num_rows==2
