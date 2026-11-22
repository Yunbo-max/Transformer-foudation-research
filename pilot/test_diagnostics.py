"""Event timing and weighted-summary checks, authored for local execution.

These checks have not been executed by the Web source reviewer.
"""
from types import SimpleNamespace
import unittest

import torch

from common import _aggregate_panel, _gate_diagnostics, generate_batch


class EventDiagnosticTests(unittest.TestCase):
    def _batch(self):
        return generate_batch(batch=1, pairs=2, queries=4, distractors=2,
                              key_space=16, seed=7)

    def test_event_positions_match_explicit_token_timeline(self):
        batch = self._batch()
        # Hand-enumerated timeline: initial facts, query, alias fact/query,
        # query, new facts/queries, then hot/cold old-fact queries.
        expected = {
            "initial_fact": [3, 6], "alias_fact": [12], "new_fact": [21, 24],
            "answer_old_pre": [9, 18], "answer_alias": [15],
            "answer_new_post": [27, 30], "answer_old_post": [33, 36],
            "answer_old_post_hot": [33], "answer_old_post_cold": [36],
        }
        self.assertEqual(batch["length"], 37)
        for name, positions in expected.items():
            with self.subTest(event=name):
                actual = batch["event_token_masks"][name][0].nonzero().flatten().tolist()
                self.assertEqual(actual, positions)
        self.assertEqual(batch["groups"]["alias"][0].nonzero().flatten().tolist(), [14])
        fact_union = (batch["event_token_masks"]["initial_fact"]
                      | batch["event_token_masks"]["alias_fact"]
                      | batch["event_token_masks"]["new_fact"])
        self.assertTrue(torch.equal(fact_union, batch["fact_token_mask"]))
        self.assertFalse(bool((fact_union & batch["answer_token_mask"]).any()))

    def test_diagnostics_observe_value_write_not_preceding_prediction(self):
        batch = self._batch()
        position = torch.arange(37, dtype=torch.float32).view(1, 1, 37)
        rho = position * .01 + torch.tensor([0., .4]).view(1, 2, 1)
        residual = position + torch.tensor([0., 100.]).view(1, 2, 1)
        before = rho.clone()
        mixer = SimpleNamespace(last_rho=rho,
            last_diagnostics={"baseline_edit_norm": residual})
        model = SimpleNamespace(blocks=[SimpleNamespace(mixer=mixer)])
        events = _gate_diagnostics(model, batch)["gate_events"]["layer0"]
        alias = events["alias_fact"]
        self.assertEqual(alias["token_count"], 1)
        self.assertEqual(alias["head_token_count"], 2)
        self.assertAlmostEqual(alias["metrics"]["rho"]["mean"], .32, places=6)
        self.assertEqual(alias["metrics"]["baseline_edit_norm"]["sum"], 124.)
        self.assertEqual(alias["metrics"]["baseline_edit_norm"]["mean"], 62.)
        self.assertEqual(alias["rho_below_half_count"], 1)
        self.assertAlmostEqual(events["answer_alias"]["metrics"]["rho"]["mean"],
                               .35, places=6)
        self.assertTrue(torch.equal(rho, before))

    def test_panel_aggregation_weights_observations_instead_of_batch_means(self):
        def row(tokens, rho_sum, below_half):
            count = tokens * 2
            return {"seq_len": 37, "gate_events": {"layer0": {"alias_fact": {
                "token_count": tokens, "head_token_count": count,
                "rho_below_half_count": below_half,
                "rho_below_half_fraction": below_half / count,
                "metrics": {"rho": {"sum": rho_sum, "count": count,
                                    "mean": rho_sum / count}},
            }}}}
        panel = _aggregate_panel([row(1, 1., 1), row(3, 1.2, 6)], pairs=2,
            distractors=2, eval_seeds=[101, 102], expected_batches=2, queries=4)
        event = panel["gate_events"]["layer0"]["alias_fact"]
        self.assertEqual(event["token_count"], 4)
        self.assertEqual(event["head_token_count"], 8)
        self.assertEqual(event["metrics"]["rho"]["count"], 8)
        self.assertAlmostEqual(event["metrics"]["rho"]["sum"], 2.2)
        self.assertAlmostEqual(event["metrics"]["rho"]["mean"], .275)
        self.assertEqual(event["rho_below_half_fraction"], 7 / 8)

    def test_detailed_events_require_observed_diagnostics(self):
        batch = self._batch()
        mixer = SimpleNamespace(last_rho=torch.ones(1, 2, 37), last_diagnostics=None)
        model = SimpleNamespace(blocks=[SimpleNamespace(mixer=mixer)])
        result = _gate_diagnostics(model, batch)
        self.assertNotIn("gate_events", result)
        self.assertEqual(result["layer0_rho_answer"], 1.)
        mixer.last_diagnostics = {"baseline_edit_norm": torch.ones(1, 2, 36)}
        with self.assertRaises(ValueError):
            _gate_diagnostics(model, batch)


if __name__ == "__main__":
    unittest.main()
