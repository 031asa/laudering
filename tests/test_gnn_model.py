"""unittest keeps the existing Python 3.8 Torch environment unchanged."""
import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run model tests in the existing d2l-zh PyTorch environment')
from gnn_features.model import GINELayer,BidirectionalGINE


class ModelTests(unittest.TestCase):
    def test_gine_hand_calculation_and_empty_edges(self):
        layer=GINELayer(2);layer.mlp=torch.nn.Identity()
        x=torch.tensor([[1.,2.],[3.,4.]])
        edges=torch.tensor([[0,0],[1,1]])
        values=torch.tensor([[.5,-3.],[1.,1.]])
        torch.testing.assert_close(layer(x,edges,values),torch.tensor([[1.,2.],[6.5,7.]]))
        torch.testing.assert_close(layer(x,edges[:,:0],values[:0]),x)

    def test_node_permutation_and_empty_history(self):
        torch.manual_seed(4);model=BidirectionalGINE(4,[4,4,4,7],width=8,dropout=0).eval()
        b=dict(x_num=torch.randn(3,63),x_cat=torch.tensor([[1],[2],[3]]),edge_index_in=torch.tensor([[0,1],[1,2]]),
            edge_index_out=torch.tensor([[1,2],[0,1]]),edge_attr_num=torch.randn(2,9),edge_attr_cat=torch.zeros(2,4,dtype=torch.long),
            edge_label_index=torch.tensor([[0,1],[2,1]]),target_base_num=torch.randn(2,8),target_base_cat=torch.zeros(2,4,dtype=torch.long),target_attr=torch.randn(2,110))
        perm=torch.tensor([2,0,1]);inverse=torch.argsort(perm);changed=dict(b)
        for key in ['x_num','x_cat']:changed[key]=b[key][perm]
        for key in ['edge_index_in','edge_index_out','edge_label_index']:changed[key]=inverse[b[key]]
        torch.testing.assert_close(model(b),model(changed))
        for key in ['edge_index_in','edge_index_out']:b[key]=b[key][:,:0]
        for key in ['edge_attr_num','edge_attr_cat']:b[key]=b[key][:0]
        self.assertTrue(torch.isfinite(model(b)).all())
        self.assertEqual(tuple(model(b).shape),(2,))


if __name__=='__main__':unittest.main()
