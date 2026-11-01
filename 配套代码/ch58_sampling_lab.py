"""Seeded categorical sampling on four fixed logits; no model generation."""
import random,json,argparse,math
from pathlib import Path
from ch44_65_lab import softmax

def categorical(prob,rng):
    if not prob or any(not math.isfinite(p) or p<0 for p in prob):raise ValueError('invalid probabilities')
    total=sum(prob)
    if total<=0:raise ValueError('empty probability support')
    u=rng.random()*total;cumulative=0.
    for i,p in enumerate(prob):
        cumulative+=p
        if u<cumulative:return i
    return max(i for i,p in enumerate(prob) if p>0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
 p=softmax([2.,1.,0.,-1.]);rng=random.Random(58);counts=[0]*4;n=100000
 for _ in range(n):counts[categorical(p,rng)]+=1
 empirical=[c/n for c in counts]
 assert max(abs(a-b) for a,b in zip(p,empirical))<.01
 result={'scope':'fixed-logit CPU sampling check','seed':58,'trials':n,'probabilities':p,'counts':counts,'empirical':empirical,'no_D_in_10_probability':(1-p[3])**10}
 s=json.dumps(result,ensure_ascii=False,indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
