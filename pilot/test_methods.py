"""Algebra, causality and backend-contract checks; no benchmark verdicts.

Generated for local execution. These tests have not been run in the Web host.
"""
import math
import unittest

import torch
import torch.nn.functional as F

from common import MemoryLM, answer_stats, generate_batch
from predicted_skip import (
    CausalDepthwiseConv, GDN2PredSkipMixer, beta_guard, edit_step, normalize_qk,
)


class PredSkipTests(unittest.TestCase):
    def test_joint_is_exact_scalar_on_complete_raw_active_edit(self):
        torch.manual_seed(2)
        state = torch.randn(5, 3, dtype=torch.double)
        key, value = torch.randn(5, dtype=torch.double), torch.randn(3, dtype=torch.double)
        erase = torch.sigmoid(torch.randn(5, dtype=torch.double))
        write = torch.sigmoid(torch.randn(3, dtype=torch.double))
        decay, rho = -.1 * torch.rand(5, dtype=torch.double), .13
        original = edit_step(state, key, value, erase, write, decay)
        scaled = edit_step(state, key, value, erase * rho, write * rho, decay)
        prior = torch.exp(decay)[:, None] * state
        self.assertTrue(torch.allclose(scaled, prior + rho * (original - prior), atol=1e-12))

    def test_write_only_breaks_a_correct_mapping_but_joint_does_not(self):
        state = torch.eye(2, dtype=torch.double)
        key = value = torch.tensor([1., 0.], dtype=torch.double)
        gate = torch.full_like(key, .6)
        joint = edit_step(state, key, value, gate * .2, gate * .2)
        write_only = edit_step(state, key, value, gate, gate * .2)
        self.assertTrue(torch.equal(joint, state))
        self.assertLess(float((write_only.T @ key)[0]), 1.)

    def test_skip_still_decays(self):
        state = torch.eye(2)
        key = value = torch.ones(2)
        decay = torch.tensor([-.1, -.2])
        out = edit_step(state, key, value, torch.zeros(2), torch.zeros(2), decay)
        self.assertTrue(torch.allclose(out, torch.diag(torch.exp(decay))))

    def test_high_cosine_does_not_certify_new_key_redundancy(self):
        state = torch.diag(torch.tensor([1., 0.], dtype=torch.double))
        query = value = torch.tensor([1., 0.], dtype=torch.double)
        key = torch.tensor([0., 1.], dtype=torch.double)
        gate = torch.full_like(key, .6)
        score = F.cosine_similarity((state.T @ query)[None], value[None]).item()
        rho = math.exp(-5 * max(score, 0))
        baseline = edit_step(state, key, value, gate, gate)
        joint = edit_step(state, key, value, gate * rho, gate * rho)
        self.assertEqual(score, 1.)
        self.assertGreater(float((joint.T @ key - value).square().sum()),
                           float((baseline.T @ key - value).square().sum()))

    def test_decay_refresh_counterexample_is_a_fixed_point(self):
        alpha, beta, rho = .99, .5, math.exp(-5)
        amplitude = rho * beta / (1 - alpha + alpha * rho * beta)
        state = torch.tensor([[amplitude]], dtype=torch.double)
        one = torch.ones(1, dtype=torch.double)
        out = edit_step(state, one, one, one * beta * rho, one * beta * rho,
                        one * math.log(alpha))
        self.assertTrue(torch.allclose(out, state, atol=1e-12))
        self.assertLess(amplitude, .26)
        self.assertGreater(beta / (1 - alpha + alpha * beta), .99)

    def test_guard_reduces_unsafe_contrast_and_retains_safe_rows(self):
        key = normalize_qk(torch.ones(1, 2))
        decay = torch.full_like(key, -.001)
        unsafe = torch.tensor([[.99, .01]])
        effective = beta_guard(key, unsafe, decay)
        self.assertFalse(torch.allclose(unsafe, effective))
        transition = (torch.eye(2) - torch.outer(key[0], effective[0] * key[0])) @ torch.diag(decay[0].exp())
        self.assertLessEqual(float(torch.linalg.matrix_norm(transition, ord=2)), 1 + 2e-5)
        safe = torch.full_like(key, .4)
        self.assertTrue(torch.equal(beta_guard(key, safe, decay), safe))
        self.assertTrue(torch.equal(beta_guard(torch.zeros_like(key), unsafe, decay), unsafe))

    def test_guard_uses_official_straight_through_gradient(self):
        key = normalize_qk(torch.ones(1, 2)).detach().requires_grad_()
        erase = torch.tensor([[.99, .01]], requires_grad=True)
        decay = torch.full_like(erase, -.001, requires_grad=True)
        probe = torch.tensor([[2., -3.]])
        (beta_guard(key, erase, decay) * probe).sum().backward()
        self.assertTrue(torch.equal(erase.grad, probe))
        self.assertIsNone(key.grad)
        self.assertIsNone(decay.grad)

    def test_reference_native_guard_is_not_silently_ignored(self):
        raw = GDN2PredSkipMixer(4, heads=1, key_dim=2, conv_size=0, guard_mode="raw")
        native = GDN2PredSkipMixer(4, heads=1, key_dim=2, conv_size=0, guard_mode="native")
        q = k = torch.ones(1, 1, 1, 2)
        v = w = torch.zeros_like(q)
        b = torch.tensor([.99, .01]).view(1, 1, 1, 2)
        g = torch.full_like(q, -.001)
        initial = torch.eye(2).view(1, 1, 2, 2)
        _, raw_state = raw._reference_chunk(q, k, v, g, b, w, initial)
        _, guarded_state = native._reference_chunk(q, k, v, g, b, w, initial)
        self.assertGreater(float(torch.linalg.matrix_norm(raw_state[0, 0], ord=2)), 1.03)
        self.assertLessEqual(float(torch.linalg.matrix_norm(guarded_state[0, 0], ord=2)), 1 + 2e-5)

    def test_official_adapter_passes_raw_qk_for_one_normalization(self):
        model = GDN2PredSkipMixer(4, heads=1, key_dim=2, conv_size=0, backend="official")
        captured = {}

        def fake_kernel(q, k, v, g, b, w, **kwargs):
            captured.update(q=q, k=k, **kwargs)
            return torch.zeros_like(v), kwargs["initial_state"]

        model._official_kernel = fake_kernel
        q = torch.tensor([2., 3.]).view(1, 1, 1, 2)
        k = q * 2
        v = b = w = torch.ones_like(q)
        g = -.01 * torch.ones_like(q)
        state = torch.zeros(1, 1, 2, 2)
        model._official_chunk(q, k, v, g, b, w, state)
        self.assertTrue(torch.equal(captured["q"], q.permute(0, 2, 1, 3)))
        self.assertTrue(torch.equal(captured["k"], k.permute(0, 2, 1, 3)))
        self.assertTrue(captured["use_qk_l2norm_in_kernel"])
        self.assertTrue(captured["beta_guard"])

    def test_short_convolution_is_causal_and_identity_initialized(self):
        torch.manual_seed(4)
        conv = CausalDepthwiseConv(3, 4)
        x = torch.randn(2, 9, 3)
        self.assertTrue(torch.allclose(conv(x), F.silu(x), atol=1e-7))
        with torch.no_grad():
            conv.conv.weight[..., 0] = .3
        changed = x.clone()
        changed[:, 5:] += 8
        self.assertTrue(torch.allclose(conv(x)[:, :5], conv(changed)[:, :5], atol=1e-7))

    def test_strict_causality_inside_and_across_chunk_boundaries(self):
        torch.manual_seed(8)
        model = GDN2PredSkipMixer(24, heads=2, key_dim=4, chunk=4, mode="joint")
        x = torch.randn(2, 11, 24)
        y, _ = model(x)
        for boundary in (1, 3, 4, 7, 8):
            changed = x.clone()
            changed[:, boundary:] += torch.randn_like(changed[:, boundary:])
            z, _ = model(changed)
            self.assertTrue(torch.allclose(y[:, :boundary], z[:, :boundary], atol=1e-6))

    def test_first_chunk_gate_is_one_and_gamma_is_per_head(self):
        torch.manual_seed(11)
        model = GDN2PredSkipMixer(12, heads=3, key_dim=4, chunk=4, gamma=.25).eval()
        _, _, state = model(torch.randn(2, 9, 12), return_final_state=True)
        self.assertEqual(tuple(model.log_gamma.shape), (3,))
        self.assertTrue(torch.allclose(model.log_gamma.exp(), torch.full((3,), .25)))
        self.assertTrue(torch.equal(model.last_rho[:, :, :4], torch.ones(2, 3, 4)))
        self.assertEqual(state.dtype, torch.float32)

    def test_reference_diagnostics_observe_raw_edit_scaling_without_changing_outputs(self):
        torch.manual_seed(18)
        model = GDN2PredSkipMixer(12, heads=2, key_dim=4, chunk=4, gamma=2.,
                                 guard_mode="raw", diagnostics=True).eval()
        x = torch.randn(2, 9, 12)
        output, _ = model(x)
        diagnostics = model.last_diagnostics
        self.assertTrue(torch.allclose(diagnostics["applied_edit_norm"],
                        diagnostics["rho"] * diagnostics["baseline_edit_norm"], atol=2e-6, rtol=2e-5))
        model.diagnostics = False
        other, _ = model(x)
        self.assertTrue(torch.equal(output, other))

    def test_new_fact_plasticity_is_explicitly_scored(self):
        batch = generate_batch(
            batch=3, pairs=4, queries=4, distractors=2, key_space=32, seed=20261121,
        )
        novel = batch["groups"]["new_post"]
        self.assertEqual(int(novel.sum().item()), 6)
        # Every genuinely new fact query contributes to the causal answer loss.
        self.assertTrue(bool(torch.all(batch["loss_mask"][novel])))
        for name in ("old_post", "old_post_hot", "old_post_cold", "alias"):
            with self.subTest(group=name):
                self.assertGreater(int(batch["groups"][name].sum()), 0)

    def test_end_to_end_all_used_parameters_have_finite_gradients(self):
        data = generate_batch(batch=2, pairs=3, queries=3, distractors=0, key_space=16)
        model = MemoryLM(data["vocab_size"], 24, 1, lambda dim, _:
                         GDN2PredSkipMixer(dim, heads=2, key_dim=4, chunk=4, mode="joint"))
        _, loss = answer_stats(model(data["tokens"]), data)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_input_validation_and_legacy_guard_mapping(self):
        for kwargs in ({"chunk": 0}, {"gamma": 0}, {"gamma": float("nan")}, {"conv_size": -1}):
            with self.assertRaises(ValueError):
                GDN2PredSkipMixer(12, **kwargs)
        self.assertEqual(GDN2PredSkipMixer(12, beta_guard=False).guard_mode, "raw")
        with self.assertRaises(ValueError):
            GDN2PredSkipMixer(12, guard_mode="native", beta_guard=False)
        with self.assertRaises(ValueError):
            GDN2PredSkipMixer(12, mode="shuffled")(torch.randn(1, 4, 12))


if __name__ == "__main__":
    unittest.main()
