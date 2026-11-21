"""Software regression fixtures for report logic, not benchmark evidence."""
import copy
import unittest
from screening import paired_interval, require_complete_matrix, screen_results


def fixture(score):
    panel = dict(complete=True, eval_seeds=[101, 102], pairs=4, distractors=2,
                 old_post_n=32, old_post_hot_n=8, old_post_cold_n=24)
    for group in ("old_post", "old_post_hot", "old_post_cold", "alias", "new_post"):
        panel[group + "_acc"] = score
        panel[group + "_sequence_acc"] = [score] * 8
    return dict(status="ok", steps_completed=10, steps_requested=10,
                config={"key_space": 128}, eval=[{}, panel])


class ScreeningTests(unittest.TestCase):
    def rows(self):
        return {f"{arm}_seed{seed}": fixture(score)
                for seed in (0, 1)
                for arm, score in (("joint", .8), ("base", .5), ("write", .6))}

    def screen(self, rows, seeds=(0, 1)):
        return screen_results(
            rows, list(seeds), candidate="joint",
            comparators={"base": .03, "write": .03},
            learning_reference="base", plasticity_references=["base"],
            min_sequences=8, draws=100)

    def test_single_seed_never_promotes(self):
        self.assertEqual(self.screen(self.rows(), (0,))["decision"], "INCONCLUSIVE")

    def test_duplicate_seeds_never_count_as_replication(self):
        self.assertEqual(self.screen(self.rows(), (0, 0))["decision"], "INCONCLUSIVE")

    def test_joint_must_beat_simple_base(self):
        rows = self.rows()
        for seed in (0, 1):
            rows[f"base_seed{seed}"] = fixture(.9)
        self.assertEqual(self.screen(rows)["decision"],
                         "NO_CLEAR_DEVELOPMENTAL_ADVANTAGE")

    def test_chance_level_is_inconclusive(self):
        rows = self.rows()
        for seed in (0, 1):
            rows[f"base_seed{seed}"] = fixture(.01)
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_key_agnostic_context_value_strategy_is_insufficient(self):
        rows = self.rows()
        for seed in (0, 1):
            rows[f"base_seed{seed}"] = fixture(.25)
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_hot_value_prior_is_not_binding_evidence(self):
        rows = self.rows()
        for seed in (0, 1):
            rows[f"base_seed{seed}"] = fixture(.65)
        for row in rows.values():
            row["eval"][1]["old_post_hot_n"] = 20
            row["eval"][1]["old_post_cold_n"] = 12
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_new_fact_plasticity_is_required(self):
        rows = self.rows()
        for seed in (0, 1):
            rows[f"joint_seed{seed}"]["eval"][1]["new_post_sequence_acc"] = [.2] * 8
            rows[f"joint_seed{seed}"]["eval"][1]["new_post_acc"] = .2
        self.assertEqual(self.screen(rows)["decision"],
                         "NO_CLEAR_DEVELOPMENTAL_ADVANTAGE")

    def test_unlearned_new_facts_remain_inconclusive(self):
        rows = self.rows()
        for row in rows.values():
            row["eval"][1]["new_post_sequence_acc"] = [.1] * 8
            row["eval"][1]["new_post_acc"] = .1
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_mismatched_evaluation_samples_block_pairing(self):
        rows = self.rows()
        rows["joint_seed0"]["eval"][1]["eval_seeds"] = [999]
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_completed_subset_is_not_full_training(self):
        rows = self.rows()
        rows["joint_seed1"]["steps_completed"] = 9
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_missing_step_counts_do_not_prove_completion(self):
        rows = self.rows()
        del rows["base_seed0"]["steps_completed"]
        del rows["base_seed0"]["steps_requested"]
        self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_malformed_sequence_scores_are_inconclusive(self):
        for values in ([.5] * 9, [float("nan")] * 8):
            rows = self.rows()
            rows["base_seed0"]["eval"][1]["old_post_sequence_acc"] = values
            self.assertEqual(self.screen(rows)["decision"], "INCONCLUSIVE")

    def test_missing_requested_control_blocks_completed_core_comparisons(self):
        rows = self.rows()
        report = self.screen(rows)
        self.assertEqual(report["decision"], "PROMISING_DEVELOPMENTAL")
        cases = [(seed, arm) for seed in (0, 1)
                 for arm in ("joint", "base", "write", "extra_control")]
        require_complete_matrix(report, rows, cases)
        self.assertEqual(report["decision"], "INCONCLUSIVE")
        self.assertEqual(report["execution_status"], "incomplete")

    def test_constant_paired_difference(self):
        result = paired_interval([.8] * 8, [.5] * 8, draws=100)
        self.assertAlmostEqual(result["mean"], .3)
        self.assertAlmostEqual(result["low"], .3)
        self.assertAlmostEqual(result["high"], .3)


if __name__ == "__main__":
    unittest.main()
