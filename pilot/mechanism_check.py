"""Controlled two-dimensional SG-Delta mechanism check (NO training).

Fix the sequence of write keys and their new-fact targets; vary only the
historical Query importance matrix. One-state, full geometric Write and Decay.
Preserve an independent formula implementation, compare it with a direct
quadratic solve and the actual SG-Delta helpers, then report old/new errors.
The historical matrices are prescribed constants, not estimated from the
four current writes. This checks neither a full mixer nor learned geometry.
Result is a DESIGN DIAGNOSTIC, not an LM or MQAR result. This revised script
has been source-reviewed but not executed in the Web host.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import sg_delta as sg

DT = torch.float64
I = torch.eye(2, dtype=DT)
Ck = torch.diag(torch.tensor([.5, .5], dtype=DT))
Cq_hot = torch.diag(torch.tensor([.9, .1], dtype=DT))
Cq_equal = Ck.clone()
S0 = torch.eye(2, dtype=DT)
k = F.normalize(torch.tensor([1., 1.], dtype=DT), dim=0)
v = torch.tensor([1.2, -.4], dtype=DT)
qhot, qcold = I[:, 0], I[:, 1]

def solve(C, mode, tau=.2, lam=1., mu=8., af=.95, aslow=.998, steps=4):
    if mode not in ("joint", "write", "decay"):
        raise ValueError("mode must be joint, write or decay")
    H = lam*I + mu*C
    Hinv = torch.linalg.solve(H, I)
    P = I - lam*Hinv
    state = S0.clone()
    helper_state = S0.clone()
    _, helper_P, _ = sg.query_geometry(C, lam=lam, mu=mu)
    write_mu = mu if mode in ("joint", "write") else 0.
    _, _, write_factor = sg.query_geometry(C, lam=lam, mu=write_mu)
    helper_direction = sg.protected_write_directions(k[None], write_factor, tau=tau)[0]
    write_precision = H if mode in ("joint", "write") else lam*I
    lhs = torch.outer(k, k) + tau*write_precision
    max_helper_error = 0.
    max_objective_error = 0.
    for _ in range(steps):
        D = af*I + (aslow-af)*P if mode in ("joint", "decay") else af*I
        previous = state
        prior = D @ state
        residual = v - prior.T @ k
        u = torch.linalg.solve(H, k) if mode in ("joint", "write") else k/lam
        direction = u/(tau + k@u)
        state = prior + direction[:,None]*residual[None,:]

        # Independent oracle: solve the quadratic normal equations directly.
        # For the joint arm this uses the original two distinct prior anchors,
        # rather than reusing D in the right-hand side.
        if mode == "joint":
            rhs = (torch.outer(k, v) + tau*lam*af*previous
                   + tau*mu*aslow*(C @ previous))
        else:
            rhs = torch.outer(k, v) + tau*(write_precision @ prior)
        objective_state = torch.linalg.solve(lhs, rhs)
        torch.testing.assert_close(state, objective_state, atol=1e-11, rtol=1e-11)
        max_objective_error = max(max_objective_error, float((state-objective_state).abs().max()))

        helper_D = (af*I + (aslow-af)*helper_P
                    if mode in ("joint", "decay") else af*I)
        helper_state = sg.sg_delta_step(helper_state, k, v, helper_D, helper_direction)
        torch.testing.assert_close(helper_state, state, atol=1e-11, rtol=1e-11)
        max_helper_error = max(max_helper_error, float((helper_state-state).abs().max()))
    hot = ((state-S0).T @ qhot).square().sum().item()
    cold = ((state-S0).T @ qcold).square().sum().item()
    new = ((state.T@k)-v).square().sum().item()
    return {"old_hot_mse":hot,"old_cold_mse":cold,
            "old_weighted_mse":.9*hot+.1*cold,
            "new_fact_mse":new,
            "helper_parity":"PASS",
            "max_helper_state_abs_error":max_helper_error,
            "max_direct_objective_abs_error":max_objective_error}

def run():
    report = {"status":"CONTROLLED_NUMERICAL_ONLY",
              "fixed_key_history":[.5,.5],
              "query_usage":[.9,.1],
              "fast_retention":.95,"slow_retention":.998,
              "sg_delta_source_sha256":hashlib.sha256(Path(sg.__file__).read_bytes()).hexdigest(),
              "verification_scope":"prescribed_2d_geometry_and_actual_helper_parity",
              "note":"Not a trained model or full mixer; Cq/Ck are prescribed, frozen historical statistics. Revised af=.95 lies inside the mixer gate range; earlier af=.98 numerical claims require rerunning.",
              "results":{}}
    for tau in (.02,.05,.1,.2,.5,1.,2.):
        arms = {
          "key_joint":solve(Ck,"joint",tau),
          "query_joint":solve(Cq_hot,"joint",tau),
          "query_write_only":solve(Cq_hot,"write",tau),
          "query_decay_only":solve(Cq_hot,"decay",tau),
          "aligned_query_joint":solve(Cq_equal,"joint",tau)}
        for metric in ("old_hot_mse","old_cold_mse","new_fact_mse"):
            assert abs(arms["key_joint"][metric]-
                       arms["aligned_query_joint"][metric])<1e-12
        report["results"][str(tau)] = arms
    print(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":run()

