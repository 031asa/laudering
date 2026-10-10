import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gnn_features import v2_features as f


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def test_contract_dimensions_and_existing_counts_not_duplicated():
    assert len(f.EXTRA_FIELDS)==28 and len(set(f.EXTRA_NAMES))==28
    assert f.EXTRA_DIM==59 and len(f.EXTRA_COLUMNS)==59
    assert all(x["nullable"] and x["transform"]=="log1p" for x in f.EXTRA_FIELDS)
    assert "v2_src_count_24h" not in f.EXTRA_NAMES
    assert "v2_dst_count_24h" not in f.EXTRA_NAMES
    assert f.EXTRA_COLUMNS[-3:]==["full_24h","full_72h","full_7d"]


def test_windows_lower_inclusive_all_timestamp_ties_excluded():
    t=np.array([0,0,f.DAY,f.DAY,3*f.DAY,7*f.DAY],dtype=np.int64)
    amount=np.array([1,3,10,20,30,40],dtype=np.int64)*1_000_000
    values,counts=f.role_features(t,amount)
    np.testing.assert_array_equal(counts,[0,0,2,2,0,0])
    assert values[2,0]==2 and values[3,0]==2
    assert values[2,1]==2.8
    f._check_reference(t,amount,np.zeros(len(t),dtype=np.int64),values,counts,np.arange(len(t)))
    stats=f.window_statistics(t,amount)
    assert stats["7d"]["count"][-1]==5  # events at exactly t-7d included
    assert stats["72h"]["count"][4]==4


def test_zero_history_zero_amount_single_support_and_missing_encoding():
    values,counts=f.role_features(np.array([0,1,2]),np.array([0,0,1_000_000]))
    assert np.isnan(values[0,0]) and values[1,0]==0
    assert np.isnan(values[1,2]) and np.isnan(values[2,2])
    raw=np.column_stack([values,values]);full=np.zeros((3,3))
    encoded=f.encode_extra(raw,full)
    assert encoded.shape==(3,59) and encoded.dtype==np.float32
    assert encoded[0,0]==0 and encoded[0,28]==0
    assert encoded[1,0]==0 and encoded[1,28]==1
    assert encoded[1,2]==0 and encoded[1,30]==0


def test_integer_order_above_double_exact_range():
    t=np.arange(6,dtype=np.int64)
    a=np.array([2**53+1,2**53+3,2**53+2,2**53+11,2**53+5,2**53+17],dtype=np.int64)
    values,counts=f.role_features(t,a)
    f._check_reference(t,a,np.zeros(len(t),dtype=np.int64),values,counts,np.arange(len(t)))
    # Small values and huge values together exercise rank->amount interpolation.
    b=np.array([0,1,9,2**60+1,2**60+5,2**60+17],dtype=np.int64)
    values,counts=f.role_features(t,b)
    f._check_reference(t,b,np.zeros(len(t),dtype=np.int64),values,counts,np.arange(len(t)))


def test_group_and_currency_separation_no_cross_group_window():
    t=np.array([0,1,2,0,1,2,0,1,2],dtype=np.int64)
    group=np.repeat([0,1,15],3)
    amount=np.array([1,2,3,100,200,300,7,8,9],dtype=np.int64)*1_000_000
    values,count=f.role_features(t,amount,group)
    np.testing.assert_array_equal(count,[0,1,2]*3)
    assert values[2,0]==1.5 and values[5,0]==150 and values[8,0]==7.5
    f._check_reference(t,amount,group,values,count,np.arange(len(t)))


@pytest.mark.parametrize("seed",[0,9,20260920])
def test_randomized_grouped_quantiles_match_independent_exact_reference(seed):
    rng=np.random.default_rng(seed)
    group=np.repeat(np.arange(12),rng.integers(1,30,size=12))
    t=rng.integers(0,11*24,size=len(group),dtype=np.int64)*f.DAY//24
    a=rng.integers(0,10_000_000,size=len(group),dtype=np.int64)
    order=np.lexsort((t,group));t,a,group=t[order],a[order],group[order]
    values,count=f.role_features(t,a,group)
    f._check_reference(t,a,group,values,count,np.arange(len(t)))


def test_future_mutation_and_same_time_order_do_not_change_earlier_features():
    t=np.array([0,1,1,2,2,3,4],dtype=np.int64)
    a=np.array([1,5,9,2,4,20,100],dtype=np.int64)*1_000_000
    before,_=f.role_features(t,a)
    changed=a.copy();changed[t>=3]=[2**60+3,0]
    after,_=f.role_features(t,changed)
    np.testing.assert_array_equal(before[t<3],after[t<3])
    swapped=a.copy();swapped[1:3]=swapped[1:3][::-1]
    values,_=f.role_features(t,swapped)
    np.testing.assert_array_equal(before[t>=2],values[t>=2])


@pytest.mark.parametrize("problem",["negative","float","time_order","group_order"])
def test_bad_inputs_fail_stop(problem):
    t=np.array([0,1,2]);a=np.array([1,2,3]);g=np.zeros(3,dtype=np.int64)
    if problem=="negative":a[0]=-1
    if problem=="float":a=a.astype(float)
    if problem=="time_order":t=np.array([1,0,2])
    if problem=="group_order":g=np.array([1,0,1])
    with pytest.raises(ValueError):f.role_features(t,a,g)


@pytest.mark.parametrize("problem",["infinity","negative","flags","shape"])
def test_encoded_extra_rejects_invalid_values(problem):
    raw=np.zeros((2,28));flags=np.zeros((2,3))
    if problem=="infinity":raw[0,0]=np.inf
    if problem=="negative":raw[0,0]=-1
    if problem=="flags":flags[0,0]=.5
    if problem=="shape":raw=raw[:,:27]
    with pytest.raises(ValueError):f.encode_extra(raw,flags)


def test_self_transfer_mask_and_coverage_are_independent():
    encoded=f.encode_extra(np.full((2,28),np.nan),np.array([[0,0,0],[1,1,1]]))
    np.testing.assert_array_equal(encoded[:,:56],0)
    np.testing.assert_array_equal(encoded[:,-3:],[[0,0,0],[1,1,1]])


def test_partitions_keep_accounts_whole_and_cover_zero_degree_nodes():
    ptr=np.array([0,0,3,3,1003,1003,1004,1004])
    parts=list(f._partitions(ptr,10))
    assert parts[0][0]==0 and parts[-1][1]==len(ptr)-1
    assert all(a[1]==b[0] for a,b in zip(parts,parts[1:]))
    assert (3,5) in parts or any(a<=3 and b>=4 for a,b in parts)


def make_store(tmp_path):
    cache=tmp_path/"cache";root=tmp_path/"sidecar";rel="transaction_date=2022-09-11/hour=00/block=0000"
    ids=np.array([5,3,4]);times=np.array([100,200,200])
    cp=cache/rel;cp.mkdir(parents=True)
    pq.write_table(pa.table({"targets":[[5],[3,4]],"cutoff_us":[100,200]}),cp/"plans.parquet")
    base_block=dict(status="verified",config_sha256="basehash",block=rel,targets=3,target_ids_sha256=f.array_sha(ids),file_sha256={"plans.parquet":f.file_sha(cp/"plans.parquet")})
    write_json(cp/"manifest.json",base_block)
    write_json(cache/"manifest.json",dict(status="completed",targets=3,config_sha256="basehash",blocks=[dict(block=rel,targets=3)]))
    config=dict(split="validation",offset=3,code_sha256=f.file_sha(Path(f.__file__)),cache_root=str(cache.resolve()),contract=f.contract(),contract_sha256=f.digest(f.contract()),base_manifest_sha256=f.file_sha(cache/"manifest.json"))
    write_json(root/"config.json",config);ch=f.file_sha(root/"config.json")
    raw=np.ones((3,28));raw[0]=np.nan
    raw[1,f.EXTRA_NAMES.index("v2_src_count_7d")]=7
    raw[2,f.EXTRA_NAMES.index("v2_dst_count_7d")]=25
    encoded=f.encode_extra(raw,np.ones((3,3)))
    data={"edge_id":ids,"cutoff_us":times,**{n:encoded[:,i] for i,n in enumerate(f.EXTRA_COLUMNS)},"src_count_7d":np.array([-1,7,1]),"dst_count_7d":np.array([-1,1,25])}
    folder=root/rel;folder.mkdir(parents=True)
    pq.write_table(pa.table(data),folder/"features.parquet")
    write_json(folder/"manifest.json",dict(status="verified",block=rel,targets=3,config_sha256=ch,base_manifest_sha256=f.file_sha(cp/"manifest.json"),target_ids_sha256=f.array_sha(ids),file_sha256=f.file_sha(folder/"features.parquet")))
    write_json(root/"pilot.json",dict(status="passed",config_sha256=ch))
    write_json(root/"count_crosscheck.json",dict(status="passed",config_sha256=ch,targets=3))
    write_json(root/"currency_verification.json",dict(status="passed",config_sha256=ch))
    write_json(root/"manifest.json",dict(status="completed",targets=3,config_sha256=ch,pilot_sha256=f.file_sha(root/"pilot.json"),count_crosscheck_sha256=f.file_sha(root/"count_crosscheck.json"),currency_verification_sha256=f.file_sha(root/"currency_verification.json"),blocks=[dict(block=rel,targets=3,manifest_sha256=f.file_sha(folder/"manifest.json"))]))
    return root,cache,rel,ids,times,encoded


def test_store_reorders_mixed_cutoffs_scalar_cutoff_support_and_copy(tmp_path):
    root,cache,rel,ids,times,expected=make_store(tmp_path)
    store=f.SidecarStore(root,cache,"validation")
    np.testing.assert_array_equal(store.block(rel,[4,5],[200,100]),expected[[2,0]])
    np.testing.assert_array_equal(store.block(rel,[3,4],200),expected[[1,2]])
    support=store.support(rel,[3,4],[200,200])
    np.testing.assert_array_equal(support["src_count_7d"],[7,1])
    np.testing.assert_array_equal(support["dst_count_7d"],[1,25])
    a=store.block(rel,[3],200);a[:]=0
    np.testing.assert_array_equal(store.block(rel,[3],200),expected[[1]])


@pytest.mark.parametrize("problem",["ids","cutoff","duplicate","unknown_block","split"])
def test_store_rejects_bad_requests(tmp_path,problem):
    root,cache,rel,ids,times,_=make_store(tmp_path)
    if problem=="split":
        with pytest.raises(ValueError):f.SidecarStore(root,cache,"train")
        return
    store=f.SidecarStore(root,cache,"validation")
    with pytest.raises(ValueError):
        if problem=="ids":store.block(rel,[10],100)
        if problem=="cutoff":store.block(rel,[3],100)
        if problem=="duplicate":store.block(rel,[3,3],200)
        if problem=="unknown_block":store.block("absent",[3],200)


@pytest.mark.parametrize("problem",["checksum","manifest","contract","incomplete","base"])
def test_store_rejects_changed_artifacts(tmp_path,problem):
    root,cache,rel,ids,times,_=make_store(tmp_path)
    if problem=="checksum":
        with (root/rel/"features.parquet").open("ab") as s:s.write(b"corrupt")
    if problem=="manifest":
        with (root/rel/"manifest.json").open("a") as s:s.write(" ")
    if problem=="contract":
        cfg=f.read(root/"config.json");cfg["contract_sha256"]="wrong";write_json(root/"config.json",cfg)
    if problem=="incomplete":
        m=f.read(root/"manifest.json");m["status"]="running";write_json(root/"manifest.json",m)
    if problem=="base":
        with (cache/"manifest.json").open("a") as s:s.write(" ")
    with pytest.raises(ValueError):
        store=f.SidecarStore(root,cache,"validation");store.block(rel,ids,times)


def test_paths_and_low_disk_stop_before_creation(tmp_path,monkeypatch):
    with pytest.raises(ValueError):f.inside(tmp_path,"../elsewhere")
    monkeypatch.setattr(f.shutil,"disk_usage",lambda _:type("D",(),{"free":19*1024**3})())
    with pytest.raises(RuntimeError,match="20 GiB"):f.ensure_space(tmp_path)


def test_only_current_v2_pause_is_honored(tmp_path):
    write_json(tmp_path/"result/gnn_tuning_v1/pause_request.json",dict(enabled=True))
    f._pause(tmp_path)
    write_json(tmp_path/"result/gnn_tuning_v2/pause_request.json",dict(enabled=False))
    f._pause(tmp_path)
    write_json(tmp_path/"result/gnn_tuning_v2/pause_request.json",dict(enabled=True))
    with pytest.raises(f.Paused):f._pause(tmp_path)


def test_completed_reuse_preserves_all_artifact_bytes(tmp_path):
    root,cache,rel,ids,times,_=make_store(tmp_path)
    paths=sorted(p for p in root.rglob("*") if p.is_file())
    before={str(p):p.read_bytes() for p in paths}
    first=f.verify_complete(root,cache,"validation")
    second=f.verify_complete(root,cache,"validation")
    assert first==second
    assert before=={str(p):p.read_bytes() for p in paths}


def test_selected_acceptance_uses_strict_history_and_self_applicability():
    arrays={"timestamp":np.array([0,0,1,1,2,3],dtype=np.int64),
            "src":np.array([0,0,1,0,0,0]),"dst":np.array([1,2,0,1,0,1]),
            "payment_currency_cat":np.array([2,3,2,2,2,2]),
            "receiving_currency_cat":np.array([2,3,2,2,2,2]),
            "paid_scaled":np.array([1,10,20,3,9,5],dtype=np.int64)*1_000_000,
            "received_scaled":np.array([1,10,20,3,9,5],dtype=np.int64)*1_000_000}
    for direction,key in [("in","dst"),("out","src")]:
        arrays[direction+"_order"]=np.lexsort((np.arange(6),arrays["timestamp"],arrays[key]))
        arrays[direction+"_ptr"]=np.r_[0,np.cumsum(np.bincount(arrays[key],minlength=3))]
    encoded,counts=f.compute_selected(arrays,0,[3,4,5],return_count24=True)
    np.testing.assert_array_equal(counts,[[1,1],[-1,-1],[2,2]])
    np.testing.assert_array_equal(encoded[1,:56],0)
    assert encoded[0,f.EXTRA_NAMES.index("v2_src_amount_q50_24h")]==np.float32(np.log1p(1))
    assert encoded[2,f.EXTRA_NAMES.index("v2_src_amount_q50_24h")]==np.float32(np.log1p(2))
    # The same-time incoming transaction and future data cannot affect row 3.
    arrays["paid_scaled"][2]=10**18
    arrays["paid_scaled"][5]=10**18
    np.testing.assert_array_equal(f.compute_selected(arrays,0,[3]),encoded[[0]])


@pytest.mark.parametrize("pause_once",[False,True])
def test_production_flow_small_fixture_resume_and_immutable_completed_reuse(tmp_path,monkeypatch,pause_once):
    project=tmp_path
    archive=project/"data/gnn_v1_train";cache=project/"data/gnn_train_cache_parquet"
    result=project/"result/gnn_train_v1";date="2022-09-01"
    spec=dict(f.SPECS["train"],targets=6,dates=[date])
    monkeypatch.setitem(f.SPECS,"train",spec)
    code=project/"scripts/gnn_features/v2_features.py"
    code.parent.mkdir(parents=True);code.write_bytes(Path(f.__file__).read_bytes())
    (code.parent/"training_state.py").write_bytes(Path(f.__file__).with_name("training_state.py").read_bytes())
    monkeypatch.setattr(f,"__file__",str(code))
    beginning=int(np.datetime64("2022-09-01T00:00:00","us").astype(np.int64))
    arrays={"timestamp":beginning+np.array([0,0,1,1,2,3],dtype=np.int64),
            "src":np.array([0,0,1,0,0,0]),"dst":np.array([1,2,0,1,0,1]),
            "payment_currency_cat":np.array([2,3,2,2,2,2]),
            "receiving_currency_cat":np.array([2,3,2,2,2,2]),
            "paid_scaled":np.array([1,10,20,3,9,5],dtype=np.int64)*1_000_000,
            "received_scaled":np.array([1,10,20,3,9,5],dtype=np.int64)*1_000_000}
    for name,values in arrays.items():
        path=archive/"history_index"/(name+".npy");path.parent.mkdir(parents=True,exist_ok=True);np.save(path,values)
    ext=arrays["src"]!=arrays["dst"]
    for direction,key in [("in","dst"),("out","src")]:
        order=np.lexsort((np.arange(6),arrays["timestamp"],arrays[key]));order=order[ext[order]]
        ptr=np.r_[0,np.cumsum(np.bincount(arrays[key][ext],minlength=3))]
        arrays[direction+"_ptr"]=ptr;arrays[direction+"_edge_ids"]=order
        folder=archive/"graph/adj_index";folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/(direction+"_ptr.npy"),ptr);np.save(folder/(direction+"_edge_ids.npy"),order)
    write_json(archive/"build.json",dict(rows=6,end_exclusive=spec["end"],data_start_us=beginning))
    write_json(archive/"metadata/vocabulary.json",dict(offset=2,values=dict(currency=["USD","EUR"])))
    folder=archive/f"graph/edges/transaction_date={date}";folder.mkdir(parents=True)
    pq.write_table(pa.table({"edge_id":np.arange(6),"payment_currency":["USD","EUR","USD","USD","USD","USD"],"receiving_currency":["USD","EUR","USD","USD","USD","USD"]}),folder/"part.parquet")
    reference,counts=f.compute_selected(arrays,beginning,np.arange(6),return_count24=True)
    folder=result/f"transaction_date={date}";folder.mkdir(parents=True)
    pq.write_table(pa.table({"edge_id":np.arange(6),"timestamp":pa.array(arrays["timestamp"],type=pa.timestamp("us")),
                             "r_src_ext_out_count_24h":pa.array(counts[:,0].astype(float),mask=counts[:,0]<0),
                             "r_dst_ext_in_count_24h":pa.array(counts[:,1].astype(float),mask=counts[:,1]<0)}),folder/"features.parquet")
    write_json(folder/"daily_run.json",dict(status="verified"))
    write_json(folder/"merge_phase.json",dict(parts=[dict(path=str(folder/"features.parquet"))]))
    write_json(cache/"config.json",dict(hashes={}))
    ch=f.file_sha(cache/"config.json");rel=f"transaction_date={date}/hour=00/block=0000"
    folder=cache/rel;folder.mkdir(parents=True)
    pq.write_table(pa.table({"targets":[[0,1],[2,3],[4],[5]],"cutoff_us":beginning+np.arange(4)}),folder/"plans.parquet")
    write_json(folder/"manifest.json",dict(status="verified",block=rel,config_sha256=ch,targets=6,target_ids_sha256=f.array_sha(np.arange(6)),file_sha256={"plans.parquet":f.file_sha(folder/"plans.parquet")}))
    write_json(cache/"manifest.json",dict(status="completed",targets=6,config_sha256=ch,edge_and_identity_tables=str(archive),feature_tables=str(result),blocks=[dict(block=rel,targets=6)]))
    reference_dir=project/"result/gnn_tuning_v1/trials/lr_only"
    pinned_paths=list((archive/"history_index").glob("*.npy"))+list((archive/"graph/adj_index").glob("*.npy"))+[archive/"build.json",archive/"metadata/vocabulary.json"]
    write_json(reference_dir/"config.json",dict(input_hashes={str(p.relative_to(project)):f.file_sha(p) for p in pinned_paths}))
    write_json(reference_dir/"training_result.json",dict(status="completed",targets=6,config_sha256=f.file_sha(reference_dir/"config.json")))
    if pause_once:
        def request_pause(**fields):
            if fields.get("stage")=="role_partition":
                write_json(project/"result/gnn_tuning_v2/pause_request.json",dict(enabled=True))
        with pytest.raises(f.Paused):f.run(project,"train",progress=request_pause)
        part_manifest=next((project/"data/gnn_v2_target_features/train/_parts").rglob("manifest.json"))
        preserved=part_manifest.read_bytes()
        write_json(project/"result/gnn_tuning_v2/pause_request.json",dict(enabled=False))
    completed=f.run(project,"train")
    if pause_once:assert part_manifest.read_bytes()==preserved
    out=project/"data/gnn_v2_target_features/train"
    store=f.SidecarStore(out,cache,"train")
    np.testing.assert_allclose(store.block(rel,np.arange(6),arrays["timestamp"]),reference,rtol=0,atol=0)
    paths=[p for p in out.rglob("*") if p.is_file() and p.name not in {"status.json","process.lock"}]
    before={str(p):p.read_bytes() for p in paths}
    assert f.run(project,"train",pilot_only=True)==completed
    assert f.run(project,"train")==completed
    assert before=={str(p):p.read_bytes() for p in paths}
