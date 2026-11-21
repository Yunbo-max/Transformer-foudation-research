"""Common synthetic MQAR benchmark and small decoder, PyTorch only.
No official NVIDIA code is redistributed. No real LM benchmark claim.
"""
import json, time, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, QUERY, FACT, PAD = 0, 1, 2, 3
SCORE_GROUPS = ("answer", "old_pre", "old_post", "old_post_hot",
                "old_post_cold", "alias", "new_post")

def generate_batch(batch=4,pairs=8,queries=8,distractors=4,alias=True,*,
                   seed=0,key_space=128,device="cpu"):
    if 2*pairs+distractors+2>key_space:
        raise ValueError("insufficient key_space for distinct facts")
    if pairs<2 or queries<2: raise ValueError("pairs and queries must >=2")
    rng=random.Random(seed)
    keys_base=4; values_base=4+key_space
    tokens=[]
    for _ in range(batch):
        keys=rng.sample(range(keys_base,keys_base+key_space),pairs+distractors+1)
        vals=rng.sample(range(values_base,values_base+key_space),pairs+distractors)
        text=[BOS]
        facts=list(zip(keys[:pairs],vals[:pairs]))
        for k,v in facts:text.extend([FACT,k,v])
        targets=[];groups=[]
        # Same number of INITIAL fact writes, intentionally skewed query usage.
        # Hot fact 0 is queried often; cold fact 1 is queried at the end.
        # Query tokens also produce K/V writes, so this does not hold the
        # complete historical Key statistics fixed across query patterns.
        query_ids=[0]+[0 if rng.random()<.80 else rng.randrange(pairs)
                       for _ in range(1,queries-1)]+[1]
        if queries>=3: query_ids[-2]=0  # guarantee a hot post-interference probe
        for qi,j in enumerate(query_ids):
            if qi==queries//2:
                for d in range(distractors):
                    text.extend([FACT,keys[pairs+d],vals[pairs+d]])
                # Crucial plasticity control: immediately query genuinely new facts.
                # This separates remembering old facts from rejecting all new writes.
                for d in range(min(2,distractors)):
                    text.extend([QUERY,keys[pairs+d],vals[pairs+d]])
                    targets.append(len(text)-1)
                    groups.append("new_post")
            text.extend([QUERY,facts[j][0],facts[j][1]])
            targets.append(len(text)-1)
            groups.append(("old_post_hot" if j==0 else "old_post_cold")
                          if qi>=queries//2 else "old_pre")
            if alias and qi==0:
                ak=keys[-1];av=facts[j][1]
                text.extend([FACT,ak,av]) # new Key, same old Value
                text.extend([QUERY,ak,av])
                targets.append(len(text)-1)
                groups.append("alias")
        tokens.append((text,targets,groups))
    T=len(tokens[0][0]);assert all(len(z[0])==T for z in tokens)
    x=torch.tensor([z[0] for z in tokens],dtype=torch.long,device=device)
    mask=torch.zeros((batch,T-1),dtype=torch.bool,device=device)
    group={g:torch.zeros_like(mask) for g in SCORE_GROUPS[1:]}
    answer_token_mask=torch.zeros((batch,T),dtype=torch.bool,device=device)
    fact_token_mask=torch.zeros_like(answer_token_mask)
    for b,(seq,targets,labels) in enumerate(tokens):
        for pos,name in zip(targets,labels):
            mask[b,pos-1]=True;group[name][b,pos-1]=True
            if name in ("old_post_hot","old_post_cold"):
                group["old_post"][b,pos-1]=True
            answer_token_mask[b,pos]=True
        for pos in range(1,len(seq)-2):
            if seq[pos]==FACT:fact_token_mask[b,pos+2]=True
    return {"tokens":x,"loss_mask":mask,"groups":group,
      "answer_token_mask":answer_token_mask,"fact_token_mask":fact_token_mask,
      "vocab_size":4+2*key_space,"length":T}

class MemoryBlock(nn.Module):
    def __init__(self,dim,mixer):
        super().__init__()
        self.norm1=nn.LayerNorm(dim);self.mixer=mixer
        self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.SiLU(),nn.Linear(4*dim,dim))
    def forward(self,x):
        y,stats=self.mixer(self.norm1(x))
        x=x+y;x=x+self.ff(self.norm2(x))
        return x,stats

class MemoryLM(nn.Module):
    def __init__(self,vocab,dim,layers,mixer_factory):
        super().__init__()
        self.embed=nn.Embedding(vocab,dim)
        self.blocks=nn.ModuleList([MemoryBlock(dim,mixer_factory(dim,i)) for i in range(layers)])
        self.norm=nn.LayerNorm(dim);self.lm_head=nn.Linear(dim,vocab,bias=False)
    def forward(self,tokens,return_stats=False):
        x=self.embed(tokens);info=[]
        for b in self.blocks:
            x,z=b(x)
            if return_stats:info.append(z)
        logits=self.lm_head(self.norm(x))
        return (logits,info) if return_stats else logits

def answer_stats(logits, batch, *, per_sequence=False):
    """Causal next-token scoring; labels/masks are used only after model forward."""
    logits = logits[:, :-1].float()
    target = batch["tokens"][:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    pred = logits.argmax(-1)
    scores = {}
    groups = [("answer", batch["loss_mask"]), *batch["groups"].items()]
    for name, mask in groups:
        n = int(mask.sum().item())
        if not n:
            continue
        correct = ((pred == target) & mask).sum()
        loss_sum = (loss.detach() * mask).sum()
        scores[name + "_acc"] = float(correct.item() / n)
        scores[name + "_nll"] = float(loss_sum.item() / n)
        scores[name + "_n"] = n
        scores[name + "_correct"] = int(correct.item())
        if per_sequence:
            counts = mask.sum(dim=1)
            valid = counts > 0
            sequence_correct = ((pred == target) & mask).sum(dim=1)[valid]
            acc = sequence_correct / counts[valid]
            nll = (loss.detach() * mask).sum(dim=1)[valid] / counts[valid]
            # Hot/cold target counts vary across sequences; retain them so
            # sequence averages and token-weighted aggregate scores are explicit.
            scores[name + "_sequence_n"] = counts[valid].cpu().tolist()
            scores[name + "_sequence_correct"] = sequence_correct.cpu().tolist()
            scores[name + "_sequence_acc"] = acc.float().cpu().tolist()
            scores[name + "_sequence_nll"] = nll.float().cpu().tolist()
    train_loss = ((loss * batch["loss_mask"]).sum()
                  / batch["loss_mask"].sum().clamp_min(1))
    return scores, train_loss

PRESETS={
 "smoke":dict(dim=32,layers=1,heads=1,head_dim=8,pairs=4,queries=4,distractors=1,batch=2,steps=2,chunk=8,key_space=32),
 "pilot":dict(dim=128,layers=2,heads=2,head_dim=16,pairs=8,queries=8,distractors=4,batch=8,steps=150,chunk=16,key_space=128),
 "stress":dict(dim=192,layers=3,heads=3,head_dim=32,pairs=16,queries=16,distractors=16,batch=4,steps=300,chunk=32,key_space=256),
 "100m-sizing":dict(dim=768,layers=12,heads=12,head_dim=64,pairs=16,queries=16,distractors=16,batch=1,steps=100,chunk=64,key_space=512),
 "340m-sizing":dict(dim=896,layers=24,heads=14,head_dim=64,pairs=16,queries=16,distractors=16,batch=1,steps=100,chunk=64,key_space=512,vocab_size=32000),
}


def precision_context(device, precision):
    device_type = torch.device(device).type
    if precision not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    if precision == "bfloat16" and device_type not in ("cpu", "cuda"):
        raise ValueError("bfloat16 autocast is supported here only on CPU/CUDA")
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                          enabled=precision == "bfloat16")


def _aggregate_panel(rows, *, pairs, distractors, eval_seeds, expected_batches,
                     queries=None):
    result = dict(pairs=pairs, distractors=distractors,
                  eval_seeds=eval_seeds, eval_batches=len(rows),
                  eval_batches_expected=expected_batches,
                  complete=len(rows) == expected_batches and bool(rows))
    if queries is not None:
        result["queries"] = queries
    if not rows:
        return result
    result["seq_len"] = rows[0]["seq_len"]
    for group in SCORE_GROUPS:
        available = [r for r in rows if group + "_n" in r]
        n = sum(r[group + "_n"] for r in available)
        if not n:
            continue
        correct = sum(r[group + "_correct"] for r in available)
        result[group + "_n"] = n
        result[group + "_correct"] = correct
        result[group + "_acc"] = correct / n
        result[group + "_nll"] = sum(
            r[group + "_nll"] * r[group + "_n"] for r in available
        ) / n
        for suffix in ("sequence_n", "sequence_correct", "sequence_acc", "sequence_nll"):
            result[group + "_" + suffix] = [
                value for row in available for value in row[group + "_" + suffix]
            ]
    metric_prefixes = tuple(g + "_" for g in SCORE_GROUPS)
    for key in rows[0]:
        if key == "seq_len" or key.startswith(metric_prefixes):
            continue
        values = [r[key] for r in rows if isinstance(r.get(key), (float, int))]
        if values:
            result[key] = sum(values) / len(values)
    return result


def _gate_diagnostics(model, batch):
    """Event labels stratify evaluation only; they never enter the predictor."""
    diagnostics = {}
    for layer, block in enumerate(model.blocks):
        mixer = block.mixer
        rho = getattr(mixer, "last_rho", None)
        if rho is None:
            continue
        for label, mask in (("answer", batch["answer_token_mask"]),
                            ("fact", batch["fact_token_mask"])):
            selected = rho.masked_select(mask[:, None, :].expand_as(rho))
            if selected.numel():
                diagnostics[f"layer{layer}_rho_{label}"] = float(selected.float().mean())
                diagnostics[f"layer{layer}_skip_fraction_{label}"] = float(
                    (selected < .5).float().mean())
        for key, value in (getattr(mixer, "last_diagnostics", None) or {}).items():
            if torch.is_tensor(value) and value.numel():
                diagnostics[f"layer{layer}_{key}_mean"] = float(value.float().mean())
    return diagnostics


def train_case(model, cfg, *, device, seed, steps, budget_seconds, out_path):
    """Bounded developer diagnostic, not a published/native MQAR benchmark."""
    if steps < 1 or budget_seconds <= 0:
        raise ValueError("steps and budget_seconds must be positive")
    eval_batches = int(cfg.get("eval_batches", 16))
    eval_batch_size = int(cfg.get("eval_batch_size", 16))
    if eval_batches < 1 or eval_batch_size < 1:
        raise ValueError("evaluation sizes must be positive")
    precision = cfg.get("precision", "float32")
    torch.manual_seed(seed)
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-3),
                                  weight_decay=.01)
    start = time.monotonic()
    deadline = start + budget_seconds
    # Reserve time for the SAME full evaluation in every arm.
    train_deadline = start + .8 * budget_seconds
    curve, status, completed, train_tokens = [], "ok", 0, 0
    for i in range(steps):
        if time.monotonic() >= train_deadline:
            status = "timeout"
            break
        batch = generate_batch(
            batch=cfg["batch"], pairs=cfg["pairs"], queries=cfg["queries"],
            distractors=cfg["distractors"], seed=seed * 1000003 + i,
            key_space=cfg["key_space"], device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, precision):
            logits = model(batch["tokens"])
            _, loss = answer_stats(logits, batch)
        if not bool(torch.isfinite(loss).item()):
            status = "nonfinite_loss"
            break
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError as exc:
            if "non-finite" not in str(exc):
                raise
            status = "nonfinite_gradient"
            break
        optimizer.step()
        completed += 1
        train_tokens += cfg["batch"] * batch["length"]
        if i % max(1, steps // 10) == 0 or i == steps - 1:
            curve.append(dict(step=completed, loss=float(loss.detach()),
                              seconds=time.monotonic() - start))
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - start
    model.eval()
    panels = [(cfg["pairs"], 0), (cfg["pairs"], cfg["distractors"]),
              (min(cfg["pairs"] * 2, cfg["key_space"] // 4), cfg["distractors"])]
    eval_rows = []
    with torch.no_grad():
        for panel_id, (pairs, ds) in enumerate(panels):
            rows, used_seeds = [], []
            for batch_id in range(eval_batches):
                if time.monotonic() >= deadline:
                    if status == "ok":
                        status = "evaluation_timeout"
                    break
                # Separate namespace from training seeds; same across paired arms.
                eval_seed = (1 << 63) + seed * 1000003 + panel_id * 10000 + batch_id
                z = generate_batch(
                    batch=eval_batch_size, pairs=pairs, queries=cfg["queries"],
                    distractors=ds, seed=eval_seed, key_space=cfg["key_space"],
                    device=device)
                with precision_context(device, precision):
                    lg, extra = model(z["tokens"], return_stats=True)
                score, _ = answer_stats(lg, z, per_sequence=True)
                score["seq_len"] = z["length"]
                aggregate = {}
                for entry in extra:
                    for key, value in entry.items():
                        if isinstance(value, (float, int)):
                            aggregate.setdefault(key, []).append(float(value))
                score.update({key: sum(values) / len(values)
                              for key, values in aggregate.items()})
                score.update(_gate_diagnostics(model, z))
                rows.append(score)
                used_seeds.append(eval_seed)
            eval_rows.append(_aggregate_panel(
                rows, pairs=pairs, distractors=ds, eval_seeds=used_seeds,
                expected_batches=eval_batches, queries=cfg["queries"]))
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - start
    if status == "ok" and elapsed_seconds > budget_seconds:
        status = "evaluation_timeout"
    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if device_type == "cuda" else None)
    result = dict(
        status=status, seed=seed, params=sum(p.numel() for p in model.parameters()),
        config=cfg, steps_requested=steps, steps_completed=completed,
        peak_allocated_vram_gb=peak, train_tokens=train_tokens,
        train_seconds=training_seconds,
        train_tokens_per_second=train_tokens / max(training_seconds, 1e-9),
        elapsed_seconds=elapsed_seconds, curve=curve, eval=eval_rows,
        validation_scope="developmental synthetic diagnostic; not native benchmark evidence",
        data="existing generated associative-recall diagnostic, NOT official MQAR/FineWeb/RULER")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
