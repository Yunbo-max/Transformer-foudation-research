"""Small mathematical unit tests; no external dataset or GitHub checkout needed."""
import unittest
import torch
from grad_grouping import accumulate_positive_target_gradients, assign_by_positive_gradient


def deterministic_kmeans(X,K,iters=12):
    """Tiny test version of original pinned upstream kmeans."""
    torch.manual_seed(1)
    mus=X[torch.randperm(len(X))[:K]].clone()
    for _ in range(iters):
        a=torch.cdist(X,mus).argmin(1)
        n=torch.bincount(a,minlength=K)
        sums=torch.zeros_like(mus).index_add_(0,a,X)
        mus=torch.where(n[:,None]>0,sums/n.clamp(min=1)[:,None],mus)
    return torch.cdist(X,mus).argmin(1)


class PositiveGradientGroupingTests(unittest.TestCase):
    def test_positive_target_ce_row_gradient(self):
        v,d=3,2
        sums=torch.zeros(v,d);counts=torch.zeros(v)
        h=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        y=torch.tensor([0,0,1])
        logits=torch.zeros(3,v)
        accumulate_positive_target_gradients(sums,counts,y,h,logits)
        expected=torch.tensor([[-2/3,-2/3],[-2/3,-2/3],[0.,0.]])
        self.assertTrue(torch.allclose(sums,expected,atol=1e-6))
        self.assertEqual(counts.tolist(),[2.,1.,0.])

    def test_direction_distinguishes_compatible_rows(self):
        R=torch.tensor([[0.,0.],[.1,0.],[1.,1.],[1.1,1.]])
        g=torch.tensor([[1.,0.],[9.,0.],[0.,2.],[0.,12.]])
        cnt=torch.ones(4)*4.
        labels,info=assign_by_positive_gradient(R,g,cnt,2,
            kmeans_fn=deterministic_kmeans,min_count=3,magnitude_weight=0.)
        self.assertEqual(labels[0],labels[1])
        self.assertEqual(labels[2],labels[3])
        self.assertNotEqual(labels[0],labels[2])
        self.assertEqual(info["reliable_words"],4)

    def test_magnitude_can_separate_same_direction(self):
        R=torch.randn(4,2)
        grad=torch.tensor([[1.,0.],[1.1,0.],[100.,0.],[110.,0.]])
        cnt=torch.full((4,),4.)
        labels,_=assign_by_positive_gradient(R,grad,cnt,2,
            kmeans_fn=deterministic_kmeans,min_count=3,magnitude_weight=1.)
        self.assertEqual(labels[0],labels[1])
        self.assertEqual(labels[2],labels[3])
        self.assertNotEqual(labels[0],labels[2])

    def test_low_count_words_fall_back_to_effective_weight(self):
        R=torch.tensor([[0.,0.],[0.,.1],[9.,9.],[9.1,9.],
                        [.05,.03],[9.05,9.04]])
        grad=torch.tensor([[1.,0.],[1.,.1],[0.,1.],[.1,1.],
                           [-2.,-2.],[-2.,-2.]])
        cnt=torch.tensor([5.,5.,5.,5.,1.,0.])
        labels,info=assign_by_positive_gradient(R,grad,cnt,2,
            kmeans_fn=deterministic_kmeans,min_count=3,magnitude_weight=0.)
        self.assertEqual(info["reliable_words"],4)
        self.assertEqual(labels[4],labels[0])
        self.assertEqual(labels[5],labels[2])

    def test_falls_back_cleanly_for_insufficient_history(self):
        R=torch.randn(5,3);g=torch.randn_like(R)
        counts=torch.tensor([1.,1.,2.,0.,1.])
        labels,info=assign_by_positive_gradient(R,g,counts,3,
            kmeans_fn=deterministic_kmeans,min_count=3)
        self.assertEqual(info["mode"],"weight_fallback")
        self.assertEqual(tuple(labels.shape),(5,))

    def test_assignment_reparameterization_keeps_logits_fixed(self):
        torch.manual_seed(11)
        R=torch.randn(32,8)
        grad=torch.randn_like(R)
        counts=torch.ones(32)*5.
        asg,_=assign_by_positive_gradient(R,grad,counts,4,
            kmeans_fn=deterministic_kmeans)
        new_center=torch.zeros(4,8).index_add_(0,asg,R)
        n=torch.bincount(asg,minlength=4)
        new_center/=n.clamp_min(1).unsqueeze(1)
        own=R-new_center[asg]
        self.assertTrue(torch.allclose(R,own+new_center[asg],atol=1e-6))
        h=torch.randn(7,8)
        self.assertTrue(torch.allclose(h@R.T,h@(own+new_center[asg]).T,atol=1e-5))


if __name__=="__main__":unittest.main()
