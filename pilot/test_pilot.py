import tempfile
import unittest
from pathlib import Path
import torch
from run import patch_gradient_regroup
from core import prepare_token_data, UPSTREAM_SHA

class RarePilotTests(unittest.TestCase):
    def test_function_preserving_reassignment(self):
        torch.manual_seed(13)
        U=torch.randn(32,8)
        C=torch.randn(4,8)
        old=torch.arange(32)%4
        R=U+C[old]
        new=(torch.arange(32)*3)%4
        center=torch.stack([R[new==i].mean(0) for i in range(4)])
        U2=R-center[new]
        self.assertTrue(torch.allclose(R,U2+center[new],atol=1e-6))
    def test_synthetic_data_expected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"upstream"
            manifest=prepare_token_data(root,8192,vocab=120)
            self.assertEqual(manifest["mode"],"synthetic-topic-zipf")
            self.assertTrue((root/"llm-per-word-weights/data/fineweb_train_000001.bin").exists())
            self.assertTrue((root/"02-muon-hypergradient-learning-rate/data/fineweb_train_000001.slice").exists())
    def test_sha_is_pinned(self):
        self.assertEqual(len(UPSTREAM_SHA),40)
if __name__=="__main__":unittest.main()
