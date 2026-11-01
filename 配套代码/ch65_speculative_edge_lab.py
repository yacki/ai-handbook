"""Exact one-position speculative correction on finite distributions."""
import argparse,json
from pathlib import Path

def correction(p,q):
 accept=[min(1.,a/b) if b>0 else 0. for a,b in zip(p,q)]
 accepted=[b*t for b,t in zip(q,accept)];z=1-sum(accepted)
 positive=[max(a-b,0.) for a,b in zip(p,q)];total=sum(positive)
 residual=[v/total for v in positive] if total>1e-14 else None
 out=[a+(z*r if residual else 0.) for a,r in zip(accepted,residual or [0.]*len(p))]
 assert max(abs(a-b) for a,b in zip(p,out))<1e-12
 return {'target':p,'draft':q,'acceptance':accept,'rejection_mass':z,'residual':residual,'output':out}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
 cases=[correction([.5,.3,.2],[.2,.5,.3]),correction([.2,.3,.5],[.2,.3,.5]),correction([1.,0.,0.],[0.,1.,0.]),correction([.6,.4],[.8,.2])]
 s=json.dumps({'scope':'finite probability identities, no neural decoding benchmark','cases':cases},indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
