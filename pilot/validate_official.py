"""Optional NVIDIA chunk_gdn2 forward+gradient parity on CUDA (not CPU-tested)."""
import argparse,json,torch
from predicted_skip import GDN2PredSkipMixer
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-root",required=True)
    p.add_argument("--tolerance",type=float,default=.15)
    a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit("CUDA required")
    torch.manual_seed(417)
    ref=GDN2PredSkipMixer(128,heads=2,key_dim=32,chunk=64,mode="joint")
    official=GDN2PredSkipMixer(128,heads=2,key_dim=32,chunk=64,
        mode="joint",backend="official",official_root=a.official_root,beta_guard=False)
    official.load_state_dict(ref.state_dict())
    ref=ref.cuda();official=official.cuda()
    x=torch.randn(2,128,128,device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        yr,_=ref(x);yo,_=official(x)
        lossr=yr.float().square().mean()
        losso=yo.float().square().mean()
    lossr.backward();losso.backward()
    err=(yr.float()-yo.float()).abs().max().item()
    gr=ref.q_proj.weight.grad;go=official.q_proj.weight.grad
    gerr=(gr-go).norm().item()/(gr.norm().item()+1e-12)
    status="PASS" if err<a.tolerance and gerr<.25 else "FAIL"
    print(json.dumps(dict(status=status,max_abs_output=err,
        relative_q_gradient_error=gerr,tolerance=a.tolerance),indent=2))
    if status!="PASS":raise SystemExit("Kernel parity failed; do not run trained comparison")
if __name__=="__main__":main()
