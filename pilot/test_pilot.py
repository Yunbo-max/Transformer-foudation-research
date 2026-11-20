import unittest
from run import ARMS, CONFIGS
from core import UPSTREAM_SHA

class PointerPilotTests(unittest.TestCase):
    def test_covers_failure_modes(self):
        names={c[0] for c in CONFIGS}
        self.assertEqual(names,{"reverse_7","forward_4","random_7","reverse_5","repeat_7"})
    def test_shared_source_not_two_papers(self):
        self.assertIn("PNB",ARMS)
        self.assertIn("BLN",ARMS)
        self.assertIn("PRD",ARMS)
        self.assertEqual(len(UPSTREAM_SHA),40)
    def test_full_uses_11_and_no_backtrack(self):
        args=" ".join(map(str,ARMS["BLN"]))
        self.assertIn("--ptr 11",args)
        self.assertIn("--pnb 3",args)
if __name__=="__main__":unittest.main()
