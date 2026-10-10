"""Read a node's exact historical features without a precomputed snapshot table."""
import argparse
import json
from pathlib import Path
import numpy as np
from .build import load_archive
from .engine import FeatureEngine
from .contract import N_FIELDS, encode_numeric

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--node-id',type=int,required=True);p.add_argument('--cutoff',required=True)
    args=p.parse_args();meta,arrays,bank,vocab=load_archive(args.archive)
    if not 0<=args.node_id<meta['node_count']:p.error('Unknown node ID')
    t=int(np.datetime64(args.cutoff,'us').astype(np.int64))
    end=int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64))
    engine=FeatureEngine(arrays,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],end)
    engine.at(t);values=engine.node(args.node_id)
    category=np.load(args.archive/'history_index/bank_cat.npy',mmap_mode='r')
    print(json.dumps(dict(node_id=args.node_id,cutoff=args.cutoff,bank_cat=int(category[args.node_id]),
        N=values,missing_reasons={k:'no_history' for k,v in values.items() if v is None},
        history_context=engine.context(),encoded_numeric=encode_numeric(values,N_FIELDS)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
