"""Scalar-valued blocked softmax identities, not GPU FlashAttention."""
import math,json,argparse
from pathlib import Path
from ch44_65_lab import softmax

def blocked(scores,values,block_size):
 m=-math.inf;den=0.;num=0.
 for start in range(0,len(scores),block_size):
  zs=scores[start:start+block_size];vs=values[start:start+block_size]
  new=max(m,max(zs));factor=0. if m==-math.inf else math.exp(m-new)
  den=den*factor+sum(math.exp(z-new) for z in zs)
  num=num*factor+sum(math.exp(z-new)*v for z,v in zip(zs,vs));m=new
 return num/den

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
 zs=[2.,1.,3.,-1.];vs=[10.,20.,30.,40.];ref=sum(p*v for p,v in zip(softmax(zs),vs));rows=[]
 for shift in [-1000.,0.,1000.]:
  for size in [1,2,3,4]:
   got=blocked([z+shift for z in zs],vs,size);err=abs(got-ref);assert err<1e-10
   rows.append({'score_shift':shift,'block_size':size,'output':got,'absolute_error':err})
 s=json.dumps({'scope':'CPU numerical identity, not GPU benchmark','reference':ref,'cases':rows},indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
