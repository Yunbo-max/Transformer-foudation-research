"""Controlled two-dimensional SG-Delta mechanism check (NO training).

Fix the sequence of write keys and their new-fact targets; vary only the
historical Query importance matrix. One-state, full geometric Write and Decay.
Tests exact reduction to Key-joint when C_q == C_k and computes old-vs-new
plasticity for several common tau values. Result is a DESIGN DIAGNOSTIC, not
an LM or MQAR result.
"""
import json
import torch
import torch.nn.functional as F

DT = torch.float64
I = torch.eye(2, dtype=DT)
Ck = torch.diag(torch.tensor([.5, .5], dtype=DT))
Cq_hot = torch.diag(torch.tensor([.9, .1], dtype=DT))
Cq_equal = Ck.clone()
S0 = torch.eye(2, dtype=DT)
k = F.normalize(torch.tensor([1., 1.], dtype=DT), dim=0)
v = torch.tensor([1.2, -.4], dtype=DT)
qhot, qcold = I[:, 0], I[:, 1]

def solve(C, mode, tau=.2, lam=1., mu=8., af=.98, aslow=.998, steps=4):
    H = lam*I + mu*C
    Hinv = torch.linalg.solve(H, I)
    P = I - lam*Hinv
    state = S0.clone()
    for _ in range(steps):
        D = af*I + (aslow-af)*P if mode in ("joint", "decay") else af*I
        prior = D @ state
        residual = v - prior.T @ k
        u = torch.linalg.solve(H, k) if mode in ("joint", "write") else k/lam
        direction = u/(tau + k@u)
        state = prior + direction[:,None]*residual[None,:]
    hot = ((state-S0).T @ qhot).square().sum().item()
    cold = ((state-S0).T @ qcold).square().sum().item()
    new = ((state.T@k)-v).square().sum().item()
    return {"old_hot_mse":hot,"old_cold_mse":cold,
            "old_weighted_mse":.9*hot+.1*cold,
            "new_fact_mse":new}

def run():
    report = {"status":"CONTROLLED_NUMERICAL_ONLY",
              "fixed_key_history":[.5,.5],
              "query_usage":[.9,.1],
              "note":"Not a trained model; known Cq is deliberately different from Ck.",
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
