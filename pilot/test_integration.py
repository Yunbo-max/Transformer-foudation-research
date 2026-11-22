"""Fail-closed adapter syntax/idempotence checks without any GPU download."""
import ast
from pathlib import Path
import tempfile
import unittest
from integrate_pointer import patch_pdg


STUB='''import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
p=object()
a = p.parse_args()
class Model:
    def forward(s):
                if a.ptr in (1, 3, 4, 5, 9, 10, 11):
                    lg0 = 0
                    A0 = torch.softmax(lg0, -1)
                    lg1 = 0
                    A1_ = torch.softmax(lg1, -1)
                if a.ptr in (6, 7, 8):
                    pass
def main():
    res={}
    step=0
    res["steps_run"] = step + 1
'''


class CoauthorSparseAdapterTests(unittest.TestCase):
    def test_patch_compiles_and_preserves_dense_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"pdg.py"
            f.write_text(STUB)
            patch_pdg(f)
            parsed=f.read_text()
            ast.parse(parsed)
            self.assertIn('a.wuke_sparse_mode == "block"',parsed)
            self.assertIn('elif a.ptr in (1, 3, 4, 5, 9, 10, 11):',parsed)
            self.assertIn('res["wuke_pointer_stats"]',parsed)
            patch_pdg(f)
            self.assertEqual(f.read_text(),parsed)

    def test_fails_when_upstream_attention_equations_change(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"pdg.py"
            f.write_text(STUB.replace("A1_ = torch.softmax(lg1, -1)","A1_ = 0"))
            with self.assertRaises(RuntimeError):
                patch_pdg(f)


if __name__=="__main__":unittest.main()
