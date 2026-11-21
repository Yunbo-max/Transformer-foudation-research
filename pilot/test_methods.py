import unittest
import torch
from sg_delta import SGDeltaMixer
from common import generate_batch,MemoryLM,answer_stats

class SGDeltaTests(unittest.TestCase):
    def test_joint_decay_retains_frequently_queried_directions(self):
        C=torch.tensor([[8.,0.],[0.,.2]],dtype=torch.double)
        H=torch.eye(2,dtype=torch.double)+C
        I=torch.eye(2,dtype=torch.double)
        P=I-torch.linalg.solve(H,I)
        D=.90*I+(.99-.90)*P
        eigen=torch.linalg.eigvalsh(D)
        self.assertTrue(((eigen>=.90-1e-10)&(eigen<=.99+1e-10)).all())
        self.assertGreater(D[0,0],D[1,1])
        k=torch.ones(2,dtype=torch.double)
        u=torch.linalg.solve(H,k)
        self.assertLess(u[0],u[1]) # same history protects the Write axis too
    def test_isotropic_limit(self):
        C=torch.randn(3,3,dtype=torch.double);C=C@C.T
        H=1.2*torch.eye(3,dtype=torch.double)+0*C
        k=torch.randn(3,dtype=torch.double)
        u=torch.linalg.solve(H,k);direction=u/(.5+k@u)
        expected=k/(.5*1.2+k@k)
        self.assertTrue(torch.allclose(direction,expected,atol=1e-12))
    def test_strict_causality(self):
        torch.manual_seed(3)
        m=SGDeltaMixer(24,heads=2,key_dim=4,chunk=4,mode="query-joint")
        x=torch.randn(2,8,24);x2=x.clone()
        x2[:,4:]+=torch.randn_like(x2[:,4:])
        y,_=m(x);z,_=m(x2)
        self.assertTrue(torch.allclose(y[:,:4],z[:,:4],atol=1e-6))
    def test_new_fact_plasticity_is_explicitly_scored(self):
        batch=generate_batch(batch=3,pairs=4,queries=4,distractors=2,
            key_space=32,seed=20261121)
        novel=batch["groups"]["new_post"]
        self.assertEqual(int(novel.sum().item()),6)
        # Every new fact query is in the causal answer loss mask.
        self.assertTrue(torch.all(batch["loss_mask"][novel]))
        self.assertGreater(int(batch["groups"]["old_post"].sum()),0)
        self.assertGreater(int(batch["groups"]["old_post_hot"].sum()),0)
        self.assertGreater(int(batch["groups"]["old_post_cold"].sum()),0)
        self.assertGreater(int(batch["groups"]["alias"].sum()),0)

    def test_end_to_end_gradients(self):
        z=generate_batch(batch=2,pairs=3,queries=3,distractors=0,key_space=16)
        model=MemoryLM(z["vocab_size"],24,1,
            lambda d,i:SGDeltaMixer(d,heads=2,key_dim=4,chunk=4,mode="query-joint"))
        _,loss=answer_stats(model(z["tokens"]),z)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None))
if __name__=="__main__":unittest.main()
