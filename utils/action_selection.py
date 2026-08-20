import hashlib, json, re
from random import Random
def _tokens(x): return set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(x)).lower().replace("_"," ")))
def _seed(*parts): return int.from_bytes(hashlib.sha256(json.dumps(parts,default=str).encode()).digest()[:8],"big")
def _score(o,k,q,et,rt):
 if k=="relation":
  r=o[0]; return len(q&_tokens(f"{r} {rt.get(r,'')}"))
 _,r,d=o; return 2*len(q&_tokens(f"{r} {rt.get(r,'')}"))+len(q&_tokens(f"{d} {et.get(d,'')}"))
def select_options(options,limit,policy,*,question,seed,step,option_kind,current_entity,entity_title,relation_title):
 if policy not in {"first","random","question-aware"}: raise ValueError(f"Unsupported max-actions policy: {policy}")
 if limit is not None and limit<1: raise ValueError("limit must be positive when provided.")
 xs=list(enumerate(options))
 if limit is None or len(xs)<=limit:return xs
 if policy=="first":return xs[:limit]
 if policy=="random":return sorted(Random(_seed(seed,question,step,option_kind,current_entity)).sample(xs,limit))
 q=_tokens(question); ranked=sorted(xs,key=lambda x:(-_score(x[1],option_kind,q,entity_title,relation_title),x[0]))
 return sorted(ranked[:limit])
