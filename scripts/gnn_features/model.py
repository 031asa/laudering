"""Two-layer bidirectional edge-aware GIN, using existing PyTorch only.

GINE aggregation follows PyG's documented formula. Native index_add avoids
requiring PyG extensions for this small CPU acceptance run.
"""
import torch
from torch import nn


class GINELayer(nn.Module):
    def __init__(self,width):
        super().__init__();self.eps=nn.Parameter(torch.zeros(()))
        self.mlp=nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))

    def forward(self,x,edge_index,edge_attr):
        src,dst=edge_index
        message=torch.relu(x[src]+edge_attr)
        aggregate=torch.zeros_like(x).index_add(0,dst,message)
        return self.mlp((1+self.eps)*x+aggregate)


class BidirectionalGINE(nn.Module):
    def __init__(self,bank_count,edge_category_sizes,node_dim=63,r_dim=110,width=64,dropout=.1):
        super().__init__()
        self.bank=nn.Embedding(bank_count,16,padding_idx=0)
        self.edge_categories=nn.ModuleList([nn.Embedding(size,8) for size in edge_category_sizes])
        self.node_encoder=nn.Sequential(nn.Linear(node_dim+16,width),nn.LayerNorm(width),nn.ReLU())
        self.history_encoder=nn.Sequential(nn.Linear(9+32,width),nn.LayerNorm(width),nn.ReLU())
        self.target_encoder=nn.Sequential(nn.Linear(8+32,32),nn.LayerNorm(32),nn.ReLU())
        self.r_encoder=nn.Sequential(nn.Linear(r_dim,64),nn.LayerNorm(64),nn.ReLU())
        self.in_layers=nn.ModuleList([GINELayer(width) for _ in range(2)])
        self.out_layers=nn.ModuleList([GINELayer(width) for _ in range(2)])
        self.fuse=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.LayerNorm(width),nn.ReLU()) for _ in range(2)])
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(4*width+32+64,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,1))

    def categories(self,categories):
        return torch.cat([emb(categories[:,i]) for i,emb in enumerate(self.edge_categories)],dim=-1)

    def forward(self,batch):
        x=self.node_encoder(torch.cat([batch['x_num'],self.bank(batch['x_cat'][:,0])],dim=-1))
        edge=self.history_encoder(torch.cat([batch['edge_attr_num'],self.categories(batch['edge_attr_cat'])],dim=-1))
        for inward,outward,fuse in zip(self.in_layers,self.out_layers,self.fuse):
            msg=torch.cat([inward(x,batch['edge_index_in'],edge),outward(x,batch['edge_index_out'],edge)],dim=-1)
            x=x+self.dropout(fuse(msg))
        src,dst=batch['edge_label_index'];u,v=x[src],x[dst]
        current=self.target_encoder(torch.cat([batch['target_base_num'],self.categories(batch['target_base_cat'])],dim=-1))
        comparison=self.r_encoder(batch['target_attr'])
        return self.head(torch.cat([u,v,torch.abs(u-v),u*v,current,comparison],dim=-1)).squeeze(-1)


def to_torch(arrays,device='cpu'):
    return {key:torch.as_tensor(value.copy(),device=device) for key,value in arrays.items()}
