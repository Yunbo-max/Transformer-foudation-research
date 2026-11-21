"""Conservative screening of the EXISTING developer diagnostic.

These records are exploratory. They never certify a native benchmark, novelty,
Gate A, or a paper claim. Independent sequence-level resampling preserves the
within-sequence grouping of repeated questions. Training-seed replication is
checked separately; the intervals are conditional on each fitted model.
"""
import math
import random


def paired_interval(left, right, *, draws=2000, seed=417):
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 40:
        raise ValueError("bootstrap draws must be an integer >= 40")
    if len(left) != len(right) or not left:
        raise ValueError("paired observations must be nonempty and equally sized")
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError("nonfinite paired observation")
    rng = random.Random(seed)
    n = len(delta)
    samples = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    return dict(mean=sum(delta) / n, low=samples[int(.025 * draws)],
                high=samples[min(draws - 1, int(.975 * draws))], sequences=n)


def require_complete_matrix(report, results, cases):
    """A requested control cannot disappear behind a completed core comparison."""
    incomplete = [f"{arm}_seed{seed}" for seed, arm in cases
                  if results.get(f"{arm}_seed{seed}", {}).get("status") != "ok"]
    report["execution_status"] = "incomplete" if incomplete else "complete"
    report["incomplete_cases"] = incomplete
    if incomplete:
        report["decision"] = "INCONCLUSIVE"
        report["reasons"].append("the requested method-by-seed matrix is incomplete")
    return report


def screen_results(results, seeds, *, candidate, comparators,
                   learning_reference, plasticity_references,
                   min_seeds=2, min_sequences=256, min_accuracy=.15,
                   draws=2000):
    report = dict(
        decision="INCONCLUSIVE", reasons=[], comparisons=[], learning_checks=[],
        scope="developmental diagnostic only; native benchmark confirmation pending",
        interval_scope="paired sequence bootstrap within seed; not a training-seed CI",
        thresholds=dict(min_seeds=min_seeds, min_sequences=min_sequences,
                        min_accuracy=min_accuracy, primary_margins=comparators,
                        plasticity_margin=-.03, bootstrap_draws=draws))
    unique_seeds = sorted(set(seeds))
    if len(unique_seeds) != len(seeds) or len(unique_seeds) < min_seeds:
        report["reasons"].append("at least two distinct complete training seeds required")
        return report
    required = set(comparators) | set(plasticity_references) | {
        candidate, learning_reference}
    panels_by_seed = {}
    for seed in unique_seeds:
        arms = {arm: results.get(f"{arm}_seed{seed}") for arm in required}
        if any(row is None or row.get("status") != "ok" for row in arms.values()):
            report["reasons"].append(f"seed {seed}: missing, failed or incomplete required arm")
            continue
        if any(not isinstance(row.get("steps_requested"), int)
               or isinstance(row.get("steps_requested"), bool)
               or row["steps_requested"] < 1
               or not isinstance(row.get("steps_completed"), int)
               or isinstance(row.get("steps_completed"), bool)
               or row.get("steps_completed") != row.get("steps_requested")
               for row in arms.values()):
            report["reasons"].append(f"seed {seed}: unmatched completed training")
            continue
        configs = [
            {k: v for k, v in row.get("config", {}).items() if k != "method"}
            for row in arms.values()
        ]
        if any(cfg != configs[0] for cfg in configs[1:]):
            report["reasons"].append(f"seed {seed}: mismatched protocol/configuration")
            continue
        if any(len(row.get("eval", [])) < 2 for row in arms.values()):
            report["reasons"].append(f"seed {seed}: missing interference panel")
            continue
        panels = {arm: row["eval"][1] for arm, row in arms.items()}
        target = panels[candidate]
        if any(not p.get("complete", False) for p in panels.values()):
            report["reasons"].append(f"seed {seed}: incomplete evaluation")
            continue
        identifiers = (target.get("eval_seeds"), target.get("pairs"),
                       target.get("distractors"))
        if (not identifiers[0] or not isinstance(identifiers[1], int)
                or identifiers[1] < 1 or not isinstance(identifiers[2], int)
                or identifiers[2] < 1) or any(
            (p.get("eval_seeds"), p.get("pairs"), p.get("distractors")) != identifiers
            for p in panels.values()
        ):
            report["reasons"].append(f"seed {seed}: mismatched or absent paired interference data")
            continue
        expected_groups = ("old_post", "old_post_hot", "old_post_cold", "alias", "new_post")
        if any(len(p.get(g + "_sequence_acc", [])) < min_sequences
               for p in panels.values() for g in expected_groups):
            report["reasons"].append(f"seed {seed}: insufficient independent sequences")
            continue
        if any(not math.isfinite(float(p.get(g + "_acc", float("nan"))))
               for p in panels.values() for g in expected_groups):
            report["reasons"].append(f"seed {seed}: nonfinite or absent score")
            continue
        if any(len(p[g + "_sequence_acc"]) != len(target[g + "_sequence_acc"])
               or not all(isinstance(value, (float, int)) and math.isfinite(value)
                          and 0 <= value <= 1 for value in p[g + "_sequence_acc"])
               for p in panels.values() for g in expected_groups):
            report["reasons"].append(f"seed {seed}: malformed or unequal paired sequence scores")
            continue
        # A predictor can emit one remembered contextual value while ignoring
        # the query key. Under the preserved hot/cold skew, always emitting the
        # first fact's value scores exactly the observed hot-target fraction.
        old_count, hot_count = target.get("old_post_n", 0), target.get("old_post_hot_n", -1)
        if (not isinstance(old_count, int) or old_count <= 0
                or not isinstance(hot_count, int) or not 0 <= hot_count <= old_count
                or any(p.get("old_post_n") != old_count
                       or p.get("old_post_hot_n") != hot_count for p in panels.values())):
            report["reasons"].append(f"seed {seed}: missing or mismatched hot/cold target counts")
            continue
        context_baseline = max(1.0 / identifiers[1], hot_count / old_count)
        floor = max(min_accuracy, context_baseline + .05)
        report["learning_checks"].append(dict(
            seed=seed, context_value_baseline=context_baseline,
            required_old_post=floor, required_new_post=min_accuracy))
        if (panels[learning_reference]["old_post_acc"] < floor
                or target["old_post_acc"] < floor):
            report["reasons"].append(
                f"seed {seed}: contextual-value floor not exceeded (required accuracy {floor:.4f})")
            continue
        if (panels[learning_reference]["new_post_acc"] < min_accuracy
                or target["new_post_acc"] < min_accuracy):
            report["reasons"].append(f"seed {seed}: new-fact acquisition below minimum diagnostic floor")
            continue
        panels_by_seed[seed] = panels
    if report["reasons"] or len(panels_by_seed) != len(unique_seeds):
        return report
    pass_all = True
    for seed, panels in panels_by_seed.items():
        target = panels[candidate]
        for arm in sorted(required):
            report.setdefault("hot_cold_scores", []).append(dict(
                seed=seed, method=arm,
                old_post_hot=panels[arm]["old_post_hot_acc"],
                old_post_cold=panels[arm]["old_post_cold_acc"],
                note="descriptive subgroup scores; not an isolated Query-versus-Key intervention"))
        for arm, margin in comparators.items():
            interval = paired_interval(
                target["old_post_sequence_acc"],
                panels[arm]["old_post_sequence_acc"], draws=draws, seed=417 + seed)
            passed = interval["mean"] >= margin and interval["low"] > 0
            pass_all &= passed
            report["comparisons"].append(dict(
                seed=seed, metric="old_post", reference=arm,
                required_margin=margin, passed=passed, **interval))
        for arm in plasticity_references:
            for group in ("alias", "new_post"):
                interval = paired_interval(
                    target[group + "_sequence_acc"],
                    panels[arm][group + "_sequence_acc"],
                    draws=draws, seed=9417 + seed)
                passed = interval["low"] >= -.03
                pass_all &= passed
                report["comparisons"].append(dict(
                    seed=seed, metric=group, reference=arm,
                    required_margin=-.03, passed=passed, **interval))
    report["decision"] = ("PROMISING_DEVELOPMENTAL" if pass_all
                          else "NO_CLEAR_DEVELOPMENTAL_ADVANTAGE")
    if pass_all:
        report["reasons"].append(
            "replicate on qualified native benchmarks; this diagnostic grants no scientific PASS")
    return report
