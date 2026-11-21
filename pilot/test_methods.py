import unittest
import torch
from predicted_skip import GDN2PredSkipMixer,edit_step
from common import generate_batch,MemoryLM,answer_stats

class PredSkipTests(unittest.TestCase):
    def test_joint_is_exact_scalar_on_complete_active_edit(self):
        torch.manual_seed(2)
        S=torch.randn(5,3,dtype=torch.double)
        k=torch.randn(5,dtype=torch.double);v=torch.randn(3,dtype=torch.double)
        b=torch.sigmoid(torch.randn(5,dtype=torch.double))
        w=torch.sigmoid(torch.randn(3,dtype=torch.double))
        g=-.1*torch.rand(5,dtype=torch.double);rho=.13
        raw=edit_step(S,k,v,b,w,g)
        scaled=edit_step(S,k,v,b*rho,w*rho,g)
        prior=torch.exp(g)[:,None]*S
        self.assertTrue(torch.allclose(scaled,prior+rho*(raw-prior),atol=1e-12))
    def test_skip_still_decays(self):
        S=torch.eye(2);k=torch.ones(2);v=torch.ones(2)
        g=torch.tensor([-.1,-.2])
        out=edit_step(S,k,v,torch.zeros(2),torch.zeros(2),g)
        self.assertTrue(torch.allclose(out,torch.diag(torch.exp(g))))
    def test_strict_causality(self):
        torch.manual_seed(8)
        m=GDN2PredSkipMixer(24,heads=2,key_dim=4,chunk=4,mode="joint")
        x=torch.randn(2,8,24);x2=x.clone()
        x2[:,4:]+=torch.randn_like(x2[:,4:])
        y,_=m(x);z,_=m(x2)
        self.assertTrue(torch.allclose(y[:,:4],z[:,:4],atol=1e-6))
    def test_end_to_end_gradients(self):
        z=generate_batch(batch=2,pairs=3,queries=3,distractors=0,key_space=16)
        model=MemoryLM(z["vocab_size"],24,1,lambda d,i:
            GDN2PredSkipMixer(d,heads=2,key_dim=4,chunk=4,mode="joint"))
        _,loss=answer_stats(model(z["tokens"]),z)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None))
if __name__=="__main__":unittest.main()
