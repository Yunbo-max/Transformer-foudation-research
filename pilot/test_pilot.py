import unittest
import torch
from run import gdn2_step, full_geometry_step, setup, repeat_answer_oracle

class ReferenceMathTests(unittest.TestCase):
    def test_gdn2_scalar_is_delta_rule(self):
        S,e,_,_,_,_=setup(7)
        k=e[0]
        v=e[2]
        beta=.33
        a=gdn2_step(S,k,v,torch.full_like(k,beta),torch.full_like(v,beta))
        b=S+beta*torch.outer(k,v-k@S)
        self.assertTrue(torch.allclose(a,b,atol=1e-12))
    def test_full_isotropic_reduction(self):
        S,e,_,_,_,_=setup(7)
        k=torch.nn.functional.normalize(e[0]+e[1],dim=0)
        v=e[4]
        out=full_geometry_step(S,k,v,torch.zeros_like(S),mu=8,lam=1,tau=1)
        expected=S+torch.outer(k,v-k@S)/(1+k@k)
        self.assertTrue(torch.allclose(out,expected,atol=1e-12))
    def test_repeated_answer_oracle_is_best(self):
        x={r["arm"]:r for r in repeat_answer_oracle(17,steps=32)}
        self.assertLess(x["oracle"]["state_drift"],x["predskip"]["state_drift"])
        self.assertLess(x["predskip"]["state_drift"],x["plain"]["state_drift"])
if __name__=="__main__":unittest.main()
