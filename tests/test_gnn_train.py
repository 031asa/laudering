from gnn_features.train import training_spec
import pytest


def test_train_excludes_validation_and_uses_exclusive_midnight():
    split={'train_members':['discovery','internal_validation'],'splits':[
        {'name':'discovery','dates':['2022-09-01'],'rows':3,'positive_rows':1},
        {'name':'internal_validation','dates':['2022-09-02'],'rows':4,'positive_rows':2},
        {'name':'validation','dates':['2022-09-03'],'rows':9,'positive_rows':9}]}
    assert training_spec(split)==(['2022-09-01','2022-09-02'],'2022-09-03T00:00:00',7,3)


def test_train_rejects_gaps():
    split={'train_members':['train'],'splits':[{'name':'train','dates':['2022-09-01','2022-09-03'],'rows':2,'positive_rows':0}]}
    with pytest.raises(ValueError,match='consecutive'):training_spec(split)
