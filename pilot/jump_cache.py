"""Online cache of compositional 2**level-hop POINTER STATES, not normal KV.

Functional-graph symbolic probe. Exact unless an optional neural edge_lookup
approximates the base edge. Edits conservatively flush all entries, so this
does NOT claim selective dependency invalidation or trained caching.
State-equivalent BF16 bytes are a logical budget, not measured GPU allocations.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import random


@dataclass
class Entry:
    dest:int
    inserted:int
    last:int
    hits:int=0


class JumpCache:
    def __init__(self,successor,slots,policy="utility",*,seed=0,
                 state_bytes=128,edge_lookup=None):
        if policy not in ("none","lru","lfu","random","level_lru","utility"):
            raise ValueError("unrecognized cache policy")
        if slots<0 or state_bytes<1:raise ValueError("invalid cache capacity")
        self.next=list(successor)
        self.edge_lookup=edge_lookup
        if any(not(0<=int(z)<len(self.next)) for z in self.next):
            raise ValueError("successor must be a total function")
        self.slots=int(slots);self.policy=policy
        self.state_bytes=state_bytes;self.cache={}
        self.rng=random.Random(seed);self.access=0
        self.popularity=defaultdict(int)
        self.hits=self.misses=self.edge_reads=self.compositions=0
        self.admissions=self.evictions=self.invalidations=0
        self.queries=0;self.peak_slots=0

    def _utility(self,key,entry):
        level=key[1]
        age=max(0,self.access-entry.last)
        if self.policy=="lru":return -age
        if self.policy=="lfu":return entry.hits
        if self.policy=="level_lru":return 2.**level/(1.+age)
        # Proxy: ONLINE access count times saved recursive work; no access
        # to future queries, ground-truth answers or evaluation feedback.
        use=math.log1p(self.popularity[key])
        return (1.+use)*min(2.**level,512.)/(1.+age/64.)

    def _put(self,key,dest):
        if self.slots==0 or self.policy=="none":return
        candidate=Entry(dest,self.access,self.access)
        if len(self.cache)>=self.slots:
            if self.policy=="random":
                victim=self.rng.choice(tuple(self.cache))
            else:
                victim=min(self.cache,key=lambda k:self._utility(k,self.cache[k]))
                if (self.policy=="utility" and
                    self._utility(key,candidate)<=self._utility(victim,self.cache[victim])):
                    return
            del self.cache[victim];self.evictions+=1
        self.cache[key]=candidate;self.admissions+=1
        self.peak_slots=max(self.peak_slots,len(self.cache))

    def _jump(self,node,level):
        key=(node,level);self.access+=1;self.popularity[key]+=1
        if level==0:
            self.edge_reads+=1
            return (self.next[node] if self.edge_lookup is None
                    else self.edge_lookup(node))
        entry=self.cache.get(key)
        if entry is not None:
            self.hits+=1;entry.hits+=1;entry.last=self.access
            return entry.dest
        self.misses+=1
        middle=self._jump(node,level-1)
        destination=self._jump(middle,level-1)
        self.compositions+=1
        self._put(key,destination)
        return destination

    def query(self,node,hops):
        if not(0<=node<len(self.next)) or hops<0:
            raise ValueError("invalid query")
        self.queries+=1
        result=node
        for level in range(hops.bit_length()-1,-1,-1):
            if (hops>>level)&1:result=self._jump(result,level)
        return result

    def update_edge(self,node,destination):
        if not(0<=node<len(self.next) and 0<=destination<len(self.next)):
            raise ValueError("invalid edge")
        if self.edge_lookup is not None and not hasattr(self.edge_lookup,"update_edge"):
            raise RuntimeError("neural edge source does not support updates")
        self.next[node]=destination
        if self.edge_lookup is not None:self.edge_lookup.update_edge(node,destination)
        self.invalidations+=len(self.cache)
        self.cache.clear() # correct but conservative; no incremental repair

    def report(self):
        return dict(policy=self.policy,queries=self.queries,
            cache_slots=self.slots,cache_hits=self.hits,
            cache_misses=self.misses,
            cache_hit_rate=self.hits/max(1,self.hits+self.misses),
            edge_reads=self.edge_reads,compositions=self.compositions,
            recompute_work=self.edge_reads+self.compositions,
            admissions=self.admissions,evictions=self.evictions,
            invalidated_entries=self.invalidations,peak_slots=self.peak_slots,
            state_equivalent_bf16_bytes=self.peak_slots*self.state_bytes)


def make_trace(num_nodes=2048,num_queries=1024,max_hops=15,seed=19,
               skew=1.2,shift_at=0.5,update_every=0):
    """Held-fixed request trace, Zipf-like popularity and drift midway."""
    if num_nodes<4 or num_queries<1 or max_hops<1:
        raise ValueError("invalid workload")
    rng=random.Random(seed)
    nodes=list(range(num_nodes));rng.shuffle(nodes)
    successor=[0]*num_nodes
    for a,b in zip(nodes,nodes[1:]+nodes[:1]):successor[a]=b
    bases=list(nodes)
    def hot(i):
        u=max(1e-9,rng.random())
        rank=min(num_nodes-1,int(u**(-1.0/max(1e-3,skew))-1))
        return bases[rank if i<int(num_queries*shift_at) else num_nodes-1-rank]
    allowed=[h for h in (1,2,3,5,7,11,15,31,63) if h<=max_hops]
    events=[]
    for i in range(num_queries):
        events.append(("query",hot(i),allowed[rng.randrange(len(allowed))]))
        if update_every and (i+1)%update_every==0:
            events.append(("update",bases[rng.randrange(num_nodes)],
                           bases[rng.randrange(num_nodes)]))
    return successor,events


def replay_trace(successor,events,slots,policy,seed=0,state_bytes=128):
    model=JumpCache(successor,slots,policy,seed=seed,state_bytes=state_bytes)
    truth=JumpCache(successor,0,"none",seed=seed,state_bytes=state_bytes)
    for typ,a,b in events:
        if typ=="update":
            model.update_edge(a,b);truth.update_edge(a,b)
        elif typ=="query":
            if model.query(a,b)!=truth.query(a,b):
                raise AssertionError("stale cached pointer after update")
        else:raise ValueError("invalid event")
    info=model.report();reference=truth.report()
    info["exact_query_accuracy"]=1.0
    info["uncached_reference_work"]=reference["recompute_work"]
    info["saved_work_fraction"]=1-info["recompute_work"]/max(1,reference["recompute_work"])
    info["interpretation"]="symbolic operations; not GPU timing or measured VRAM"
    return info
