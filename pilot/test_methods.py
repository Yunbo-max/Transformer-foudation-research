"""Mathematical regression tests for SG-Delta.

These tests are provided for local execution. Source review and syntax parsing
do not establish that this revised suite has passed.
"""
import unittest

import torch
import torch.nn.functional as F

from common import MemoryLM, answer_stats, generate_batch
from sg_delta import (
    SGDeltaMixer,
    protected_write_directions,
    query_geometry,
    sg_delta_step,
    update_query_covariance,
)


class SGDeltaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def _covariance(self, shape, key_dim):
        observations = torch.randn(*shape, key_dim, key_dim, dtype=torch.double)
        C = observations @ observations.transpose(-1, -2)
        return C / C.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]

    def test_joint_step_is_exact_unified_objective_minimizer(self):
        """Independent oracle solves the objective normal equations directly."""
        B, H, K, V = 2, 2, 4, 3
        C = self._covariance((B, H), K)
        state = torch.randn(B, H, K, V, dtype=torch.double)
        key = torch.randn(B, H, K, dtype=torch.double)
        value = torch.randn(B, H, V, dtype=torch.double)
        fast = 0.88 + 0.03 * torch.rand(B, H, dtype=torch.double)
        slow = 0.97 + 0.02 * torch.rand(B, H, dtype=torch.double)
        eye = torch.eye(K, dtype=torch.double).expand(B, H, K, K)

        for lam, mu, tau in ((1.0, 8.0, 1.0), (0.3, 2.7, 0.2), (1.4, 0.0, 3.0)):
            with self.subTest(lam=lam, mu=mu, tau=tau):
                precision, P, factor = query_geometry(C, lam=lam, mu=mu)
                decay = fast[..., None, None] * eye + (slow - fast)[..., None, None] * P
                direction = protected_write_directions(key.unsqueeze(-2), factor, tau=tau).squeeze(-2)
                actual = sg_delta_step(state, key, value, decay, direction)

                lhs = key.unsqueeze(-1) * key.unsqueeze(-2) + tau * precision
                rhs = (key.unsqueeze(-1) * value.unsqueeze(-2)
                       + tau * lam * fast[..., None, None] * state
                       + tau * mu * slow[..., None, None] * (C @ state))
                expected = torch.linalg.solve(lhs, rhs)
                torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)

                prediction_error = (actual * key.unsqueeze(-1)).sum(-2) - value
                gradient = (key.unsqueeze(-1) * prediction_error.unsqueeze(-2)
                            + tau * lam * (actual - fast[..., None, None] * state)
                            + tau * mu * (C @ (actual - slow[..., None, None] * state)))
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-11, rtol=0)

    def test_residual_contraction_uses_decayed_state(self):
        C = self._covariance((), 4)
        state = torch.randn(4, 3, dtype=torch.double)
        key = torch.randn(4, dtype=torch.double)
        value = torch.randn(3, dtype=torch.double)
        lam, mu, tau = 0.7, 3.0, 0.4
        precision, P, factor = query_geometry(C, lam=lam, mu=mu)
        decay = 0.9 * torch.eye(4, dtype=torch.double) + 0.08 * P
        direction = protected_write_directions(key[None], factor, tau=tau)[0]
        updated = sg_delta_step(state, key, value, decay, direction)
        before_error = value - (decay @ state).T @ key
        after_error = value - updated.T @ key
        r = key @ torch.linalg.solve(precision, key)
        torch.testing.assert_close(after_error, tau / (tau + r) * before_error, atol=1e-12, rtol=1e-12)

    def test_full_covariance_eigenspaces_control_both_operations(self):
        rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.double)
        eigenvalues = torch.tensor([0.8, 0.2], dtype=torch.double)
        C = rotation @ torch.diag(eigenvalues) @ rotation.T
        _, P, factor = query_geometry(C, lam=1.0, mu=8.0)
        decay = 0.90 * torch.eye(2, dtype=torch.double) + 0.09 * P
        transformed = rotation.T @ decay @ rotation
        expected = 0.90 + 0.09 * 8 * eigenvalues / (1 + 8 * eigenvalues)
        torch.testing.assert_close(transformed, torch.diag(expected), atol=1e-12, rtol=1e-12)
        self.assertGreater(float(transformed[0, 0]), float(transformed[1, 1]))
        self.assertTrue(bool((torch.linalg.eigvalsh(decay) >= 0.9 - 1e-12).all()))
        self.assertTrue(bool((torch.linalg.eigvalsh(decay) <= 0.99 + 1e-12).all()))
        key = rotation @ F.normalize(torch.ones(2, dtype=torch.double), dim=0)
        direction = protected_write_directions(key[None], factor)[0]
        coordinates = rotation.T @ direction
        self.assertLess(float(coordinates[0]), float(coordinates[1]))

    def test_isotropic_limit_for_nonunit_keys_and_zero_key(self):
        C = self._covariance((), 3)
        keys = torch.randn(5, 3, dtype=torch.double)
        keys[0] = 0
        lam, tau = 1.2, 0.5
        _, P, factor = query_geometry(C, lam=lam, mu=0)
        direction = protected_write_directions(keys, factor, tau=tau)
        expected = keys / (tau * lam + keys.square().sum(-1, keepdim=True))
        torch.testing.assert_close(direction, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(P, torch.zeros_like(P), atol=1e-12, rtol=0)
        self.assertTrue(bool(torch.isfinite(direction).all()))

    def test_joint_mixer_reduces_to_isotropic_when_mu_zero(self):
        joint = SGDeltaMixer(16, heads=2, key_dim=3, value_dim=5, chunk=4, mu=0).double()
        isotropic = SGDeltaMixer(16, heads=2, key_dim=3, value_dim=5, chunk=4, mode="isotropic").double()
        isotropic.load_state_dict(joint.state_dict())
        x = torch.randn(2, 11, 16, dtype=torch.double)
        actual, _ = joint(x)
        expected, _ = isotropic(x)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

    def test_chunk_covariance_matches_tokenwise_ema(self):
        q = F.normalize(torch.randn(2, 3, 9, 4, dtype=torch.double), dim=-1)
        initial = torch.zeros(2, 3, 4, 4, dtype=torch.double)
        for ema in (0.0, 0.8, 0.98):
            with self.subTest(ema=ema):
                expected = initial.clone()
                for t in range(q.shape[-2]):
                    qt = q[:, :, t]
                    expected = ema * expected + (1 - ema) * qt.unsqueeze(-1) * qt.unsqueeze(-2)
                actual = update_query_covariance(initial, q[:, :, :4], ema=ema)
                actual = update_query_covariance(actual, q[:, :, 4:], ema=ema)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                self.assertTrue(bool((torch.linalg.eigvalsh(actual) >= -1e-12).all()))
                trace = actual.diagonal(dim1=-2, dim2=-1).sum(-1)
                torch.testing.assert_close(trace, torch.full_like(trace, 1 - ema**9), atol=1e-12, rtol=1e-12)

    def test_statistics_detachment_is_explicit(self):
        q = torch.randn(1, 1, 5, 3, dtype=torch.double, requires_grad=True)
        C = torch.zeros(1, 1, 3, 3, dtype=torch.double)
        detached = update_query_covariance(C, q, detach_statistics=True)
        differentiable = update_query_covariance(C, q, detach_statistics=False)
        self.assertFalse(detached.requires_grad)
        self.assertTrue(differentiable.requires_grad)
        torch.testing.assert_close(detached, differentiable)
        differentiable.square().sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_strict_causality_inside_and_across_chunks(self):
        for conv_size in (0, 4):
            mixer = SGDeltaMixer(24, heads=2, key_dim=4, chunk=4, conv_size=conv_size).double()
            x = torch.randn(2, 11, 24, dtype=torch.double)
            reference, _ = mixer(x)
            for cut in (1, 3, 4, 5, 9):
                with self.subTest(conv_size=conv_size, cut=cut):
                    changed = x.clone()
                    changed[:, cut:] += 2 * torch.randn_like(changed[:, cut:])
                    output, _ = mixer(changed)
                    torch.testing.assert_close(output[:, :cut], reference[:, :cut], atol=1e-11, rtol=1e-11)

    def test_fixed_geometry_transition_contracts_in_H_norm(self):
        C = self._covariance((), 4)
        precision, P, factor = query_geometry(C)
        key = torch.randn(4, dtype=torch.double)
        direction = protected_write_directions(key[None], factor)[0]
        slow = 0.99
        decay = 0.90 * torch.eye(4, dtype=torch.double) + (slow - 0.90) * P
        a = torch.randn(4, 3, dtype=torch.double)
        b = torch.randn_like(a)
        value = torch.randn(3, dtype=torch.double)
        delta_before = a - b
        delta_after = sg_delta_step(a, key, value, decay, direction) - sg_delta_step(b, key, value, decay, direction)
        energy_before = (delta_before * (precision @ delta_before)).sum()
        energy_after = (delta_after * (precision @ delta_after)).sum()
        self.assertLessEqual(float(energy_after), float(slow**2 * energy_before) + 1e-11)

    def test_low_precision_inputs_promote_core_math_and_preserve_double(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                C = self._covariance((), 3).to(dtype)
                keys = torch.randn(2, 3).to(dtype)
                state = torch.randn(3, 2).to(dtype)
                value = torch.randn(2).to(dtype)
                H, P, factor = query_geometry(C)
                directions = protected_write_directions(keys, factor)
                decay = 0.90 * torch.eye(3, dtype=P.dtype) + 0.09 * P
                updated = sg_delta_step(state, keys[0], value, decay, directions[0])
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                for tensor in (H, P, factor, directions, updated):
                    self.assertEqual(tensor.dtype, expected_dtype)
                    self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_autocast_cannot_downcast_geometry_or_recurrence(self):
        C = self._covariance((), 3).float()
        key = torch.randn(3)
        value = torch.randn(2)
        state = torch.randn(3, 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            H, P, factor = query_geometry(C)
            direction = protected_write_directions(key[None], factor)[0]
            decay = 0.90 * torch.eye(3) + 0.09 * P
            updated = sg_delta_step(state, key, value, decay, direction)
            next_covariance = update_query_covariance(C, key[None])
        for tensor in (H, P, factor, direction, updated, next_covariance):
            self.assertEqual(tensor.dtype, torch.float32)

    def test_dimension_scaling_is_explicit_and_default_is_raw(self):
        lam, mu = 1.2, 0.7
        for K in (4, 16):
            C = torch.eye(K, dtype=torch.double) / K
            raw = SGDeltaMixer(16, heads=1, key_dim=K, lam=lam, mu=mu)
            scaled = SGDeltaMixer(16, heads=1, key_dim=K, lam=lam, mu=mu, geometry_scale="dimension")
            self.assertEqual(raw.effective_mu, mu)
            _, P_raw, _ = query_geometry(C, lam=lam, mu=raw.effective_mu)
            _, P_scaled, _ = query_geometry(C, lam=lam, mu=scaled.effective_mu)
            torch.testing.assert_close(P_raw, torch.eye(K, dtype=torch.double) * mu / (lam * K + mu))
            torch.testing.assert_close(P_scaled, torch.eye(K, dtype=torch.double) * mu / (lam + mu))

    def test_invalid_parameters_fail_instead_of_changing_the_method(self):
        for kwargs in ({"lam": 0}, {"tau": 0}, {"mu": -1}, {"ema": 1},
                       {"mu": float("inf")}, {"key_dim": 0}, {"value_dim": 0},
                       {"geometry_scale": "automatic"}, {"conv_size": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SGDeltaMixer(16, **kwargs)
        invalid_covariance = -2 * torch.eye(2, dtype=torch.double)
        with self.assertRaises(RuntimeError):
            query_geometry(invalid_covariance, lam=1, mu=1)

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

    def test_end_to_end_gradients(self):
        z = generate_batch(batch=2, pairs=3, queries=3, distractors=0, key_space=16)
        for detach in (True, False):
            with self.subTest(detach_statistics=detach):
                model = MemoryLM(
                    z["vocab_size"], 24, 1,
                    lambda d, i: SGDeltaMixer(d, heads=2, key_dim=4, chunk=4, detach_statistics=detach),
                )
                _, loss = answer_stats(model(z["tokens"]), z)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                mixer = model.blocks[0].mixer
                for module in (mixer.q_proj, mixer.k_proj, mixer.v_proj, mixer.output,
                               mixer.fast_gate, mixer.slow_gate):
                    self.assertIsNotNone(module.weight.grad)
                    self.assertTrue(bool(torch.isfinite(module.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()
