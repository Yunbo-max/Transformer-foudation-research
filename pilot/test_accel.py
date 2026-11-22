"""CPU tests of sparse Pointer math, causal routing and cache correctness."""
import unittest
import torch
from pointer_accel import pointer_lookup
from jump_cache import JumpCache,make_trace,replay_trace
from run_accel import crossed_pilot


class SparseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.q=torch.randn(2,17,8,requires_grad=True)
        self.k=torch.randn(2,17,8,requires_grad=True)
        self.v=torch.randn(2,17,5,requires_grad=True)
        self.h=torch.randn(2,17,9)

    def test_dense_and_full_local_block_match(self):
        dense,_=pointer_lookup(self.q,self.k,self.v,mode="dense",
            history=self.h,historical_keys=self.h,penalty=2.)
        sparse,_=pointer_lookup(self.q,self.k,self.v,mode="block",
            block_size=17,blocks=1,
            history=self.h,historical_keys=self.h,penalty=2.)
        self.assertTrue(torch.allclose(dense,sparse,atol=1e-6))

    def test_causal_future_key_change_does_not_change_prefix(self):
        base,_=pointer_lookup(self.q,self.k,self.v,mode="block",
            block_size=4,blocks=2,chunk=5)
        modified=self.k.detach().clone()
        modified[:,13:]*=1000.
        later,_=pointer_lookup(self.q,modified,self.v,mode="block",
            block_size=4,blocks=2,chunk=5)
        self.assertTrue(torch.allclose(base[:,:13],later[:,:13],atol=1e-6))

    def test_hash_route_is_causal_and_deterministic(self):
        first,meta=pointer_lookup(self.q,self.k,self.v,mode="hash",
            hash_bits=6,hash_probes=2,hash_bucket=16,local_window=4,chunk=5)
        again,_=pointer_lookup(self.q,self.k,self.v,mode="hash",
            hash_bits=6,hash_probes=2,hash_bucket=16,local_window=4,chunk=5)
        self.assertTrue(torch.allclose(first,again,atol=1e-6))
        changed=self.k.detach().clone()
        changed[:,13:]*=1000
        after,_=pointer_lookup(self.q,changed,self.v,mode="hash",
            hash_bits=6,hash_probes=2,hash_bucket=16,local_window=4,chunk=5)
        self.assertTrue(torch.allclose(first[:,:13],after[:,:13],atol=1e-6))
        self.assertLess(meta["scoring_pairs"],2*17*17)

    def test_gradients_and_nonquadratic_candidate_count(self):
        out,stats=pointer_lookup(self.q,self.k,self.v,mode="block",
            block_size=4,blocks=1,chunk=7)
        self.assertLess(stats["scoring_pairs"],2*17*17)
        out.square().mean().backward()
        for t in (self.q,self.k,self.v):
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())

    def test_cross_attention_and_history(self):
        q=self.q[:,-1:]
        h=self.h[:,-1:]
        pos=torch.tensor([17])
        dense,_=pointer_lookup(q,self.k,self.v,mode="dense",
            history=h,historical_keys=self.h,penalty=2.,
            query_positions=pos)
        block,_=pointer_lookup(q,self.k,self.v,mode="block",
            history=h,historical_keys=self.h,penalty=2.,
            query_positions=pos,block_size=17,blocks=1)
        self.assertTrue(torch.allclose(dense,block,atol=1e-6))


class CacheTests(unittest.TestCase):
    def test_all_policies_exact_with_edge_updates(self):
        succ,trace=make_trace(64,80,seed=15,update_every=20)
        for policy in ("none","lru","lfu","random","level_lru","utility"):
            with self.subTest(policy=policy):
                result=replay_trace(succ,trace,8,policy,seed=3)
                self.assertEqual(result["exact_query_accuracy"],1.)
                self.assertLessEqual(result["peak_slots"],8)

    def test_recompute_and_invalidation(self):
        model=JumpCache([1,2,3,0],2)
        self.assertEqual(model.query(0,3),3)
        model.update_edge(1,1)
        self.assertEqual(model.query(0,3),1)

    def test_crossed_design_runs_four_real_strategies(self):
        rows=crossed_pilot(torch.device("cpu"),10,12,6,4,1,.25)
        self.assertEqual({(r["sparse"],r["cache"]) for r in rows},
            {(a,b) for a in ("dense","block","hash") for b in ("none","utility")})
        for row in rows:
            self.assertGreaterEqual(row["exact_answer_accuracy"],0.)
            self.assertLessEqual(row["exact_answer_accuracy"],1.)


if __name__=="__main__":unittest.main()
