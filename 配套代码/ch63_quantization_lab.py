"""FP32 simulation of per-matrix INT8 rounding on shared tiny decoder.
No packed storage, low-bit kernel, 195M conversion, or performance measurement.
"""
import argparse,json,copy
from pathlib import Path
import torch
from torch import nn
from book_transformer import BookTransformer,tiny_config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
 torch.set_num_threads(1);torch.manual_seed(63)
 model=BookTransformer(tiny_config()).eval();rounded=copy.deepcopy(model)
 quantized=0;seen=set()
 with torch.no_grad():
  for module in rounded.modules():
   if isinstance(module,nn.Linear) and id(module.weight) not in seen:
    seen.add(id(module.weight));w=module.weight;scale=w.abs().max()/127
    if scale.item()>0:w.copy_((w/scale).round().clamp(-127,127)*scale)
    quantized+=w.numel()
 ids=torch.tensor([[1,3,5,7,9,11,13,15]]);labels=torch.cat((ids[:,1:],torch.tensor([[2]])),1)
 with torch.inference_mode():
  base=model(ids,labels);candidate=rounded(ids,labels)
  err=(base['logits']-candidate['logits']).abs().max().item()
  same=(base['logits'].argmax(-1)==candidate['logits'].argmax(-1)).sum().item()
 w=[-1.,-.4,.2,.9];x=[2.,-1.,.5,3.];scale=max(map(abs,w))/127
 codes=[round(v/scale) for v in w];restored=[c*scale for c in codes]
 result={'scope':'tiny numerical simulation, not compressed deployment','torch_version':torch.__version__,
   'seed':63,'simulated_quantized_parameters':quantized,'baseline_loss':base['loss'].item(),
   'rounded_loss':candidate['loss'].item(),'max_abs_logit_difference':err,
   'equal_argmax_positions':same,'positions':8,'original_dot':sum(v*t for v,t in zip(w,x)),
   'restored_dot':sum(v*t for v,t in zip(restored,x)),
   'outlier_scale':10/127,'outlier_codes':[round(v/(10/127)) for v in [.01,.02,.03,10.]]}
 s=json.dumps(result,ensure_ascii=False,indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
