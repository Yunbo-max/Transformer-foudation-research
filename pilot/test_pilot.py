import tempfile
import unittest
from pathlib import Path
from run import patch_confidence_hold
from core import UPSTREAM_SHA

class MuonPilotTests(unittest.TestCase):
    def test_patches_are_fail_closed(self):
        s="""a = p.parse_args()
curve, slog, loss_ema, diverged = [], [], None, False
mu.adapt = a.adapt_start <= step < aend and (a.freeze_at < 0 or step < a.freeze_at)
"""
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"ss.py";p.write_text(s)
            patch_confidence_hold(p)
            text=p.read_text()
            self.assertIn("dh_patience",text)
            self.assertIn("dh_counter",text)
            self.assertIn("mu.adapt = a.adapt_start",text)
            # Idempotence: no duplicate argparse options.
            patch_confidence_hold(p)
            self.assertEqual(p.read_text().count('p.add_argument("--dh_patience"'),1)
    def test_reference_commit_pinned(self):
        self.assertEqual(len(UPSTREAM_SHA),40)
if __name__=="__main__":unittest.main()
