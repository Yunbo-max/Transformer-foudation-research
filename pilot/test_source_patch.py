"""Ensures the pinned-grid textual adapter remains syntactically sound."""
import ast
from pathlib import Path
import tempfile
import unittest
from run import patch_gradient_regroup


PINNED_SHAPED_STUB='''import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import argparse
p=argparse.ArgumentParser()
a = p.parse_args()
class Model(nn.Module):
    def __init__(s):
        s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)
    def H(s):
        return s.head
    def recluster(s):
        R = s.head + s.Ch[s.ah]; K = s.Ch.shape[0]
        asg = kmeans(R, K)
        cm = torch.zeros(K, R.shape[1])
        s.ah.copy_(asg); s.Ch.copy_(cm); s.head.copy_(R - cm[asg]); s.Clast = cm.clone()
    def nll(s, tg, h, lg):
        return F.cross_entropy(lg, tg, reduction="none")
def main():
        for o in opts: o.step()
    if a.save:
        pass
'''


class UpstreamAdapterTests(unittest.TestCase):
    def test_function_preserving_adapter_compiles_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"grid.py"
            path.write_text(PINNED_SHAPED_STUB)
            patch_gradient_regroup(path)
            patched=path.read_text()
            ast.parse(patched)
            self.assertIn("s.Clast = cm.clone()",patched)
            self.assertIn('assert torch.allclose(s.H(),R',patched)
            self.assertIn('positive_dir_mag',patched)
            self.assertIn('accumulate_positive_target_gradients(',patched)
            patch_gradient_regroup(path)
            self.assertEqual(path.read_text(),patched)

    def test_refuse_unrecognized_prior_patch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"grid.py"
            path.write_text(PINNED_SHAPED_STUB.replace(
                "a = p.parse_args()", 'p.add_argument("--group_on",default="weights")\na = p.parse_args()'))
            with self.assertRaises(RuntimeError):
                patch_gradient_regroup(path)


if __name__=="__main__":unittest.main()
