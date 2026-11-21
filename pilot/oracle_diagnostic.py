"""Analytic counterexample: matching previous read ≠ current Key already stored.
No labels enter the actual Predicted-Skip model. This is a mechanism diagnostic.
"""
import json,torch
from predicted_skip import edit_step

def run():
    state=torch.eye(2,dtype=torch.float64)
    previous_query=torch.tensor([1.,0.],dtype=torch.float64)
    new_key=torch.tensor([0.,1.],dtype=torch.float64)
    value=previous_query.clone()
    old_read=state.T@previous_query
    cos=float(torch.nn.functional.cosine_similarity(old_read[None],value[None]))
    rho=float(torch.exp(torch.tensor(-5*max(0,cos))))
    beta=.6;b=torch.full_like(new_key,beta)
    w=torch.full_like(value,beta)
    options={
        "baseline":edit_step(state,new_key,value,b,w),
        "write_only":edit_step(state,new_key,value,b,w*rho),
        "joint":edit_step(state,new_key,value,b*rho,w*rho),
    }
    report={name:{
        "new_key_target_mse":float((S.T@new_key-value).square().sum()),
        "old_key_mse":float((S.T@previous_query-old_read).square().sum())
    } for name,S in options.items()}
    result=dict(previous_read_cosine=cos,rho=rho,arms=report,
      warning="Repeated content may still be a new Key-Value association; skip can hurt.")
    print(json.dumps(result,indent=2));return result
if __name__=="__main__":run()
