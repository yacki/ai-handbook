"""CPU numerical checks, not an optimized inference server.
Uses the shared tiny decoder; no pretrained model, padding, or multi-document batch.
The list/torch.cat cache favors readability and is not a GPU performance recipe.
"""
import argparse,json,math
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from book_transformer import BookTransformer,tiny_config

def rope_offset(x,past,theta):
    t,dh=x.shape[-2:]
    inv=theta**(-torch.arange(0,dh,2,device=x.device,dtype=torch.float32)/dh)
    angle=torch.arange(past,past+t,device=x.device,dtype=torch.float32)[:,None]*inv
    c,s=angle.cos().to(x.dtype),angle.sin().to(x.dtype)
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)

def cached_forward(model,ids,caches=None):
    assert ids.ndim==2 and ids.shape[0]==1
    if caches is None:caches=[None]*len(model.blocks)
    past=0 if caches[0] is None else caches[0][0].shape[-2]
    b,t=ids.shape
    assert past+t<=model.config.max_seq_len
    x=model.embedding(ids);new_cache=[]
    for block,old in zip(model.blocks,caches):
        a=block.attn;z=block.attn_norm(x)
        def split(proj):return proj(z).view(b,t,a.h,a.dh).transpose(1,2)
        q=rope_offset(split(a.q_proj),past,a.theta)
        k=rope_offset(split(a.k_proj),past,a.theta)
        v=split(a.v_proj)
        if old is not None:
            k=torch.cat((old[0],k),dim=-2);v=torch.cat((old[1],v),dim=-2)
        allowed=torch.arange(k.shape[-2])[None,:]<=torch.arange(past,past+t)[:,None]
        scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(a.dh)
        probs=scores.masked_fill(~allowed,float('-inf')).softmax(-1)
        y=(probs.to(v.dtype)@v).transpose(1,2).contiguous().view(b,t,-1)
        x=x+a.out_proj(y);x=x+block.ffn(block.ffn_norm(x))
        new_cache.append((k,v))
    return model.lm_head(model.final_norm(x)),new_cache

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(5962)
    model=BookTransformer(tiny_config()).eval()
    ids=torch.tensor([[1,3,5,7,9,11,13,15]])
    with torch.inference_mode():
        full=model(ids)['logits']
        first,caches=cached_forward(model,ids[:,:4]);pieces=[first]
        for pos in range(4,8):
            logits,caches=cached_forward(model,ids[:,pos:pos+1],caches);pieces.append(logits)
        combined=torch.cat(pieces,dim=1)
        error=(full-combined).abs().max().item()
        assert error<1e-5
        cache_bytes=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in caches)
        assert cache_bytes==2*2*1*4*8*8*4
        sdpa_model=BookTransformer(replace(tiny_config(),attention_backend='sdpa')).eval()
        sdpa_model.load_state_dict(model.state_dict())
        sdpa_error=(model(ids)['logits']-sdpa_model(ids)['logits']).abs().max().item()
        assert sdpa_error<1e-5
        q=torch.randn(1,12,3,64);k=torch.randn(1,4,5,64);v=torch.randn(1,4,5,64)
        kr=k.repeat_interleave(3,dim=1);vr=v.repeat_interleave(3,dim=1)
        via_repeat=F.scaled_dot_product_attention(q,kr,vr,dropout_p=0.)
        per_group=torch.cat([F.scaled_dot_product_attention(q[:,i*3:(i+1)*3],k[:,i:i+1].expand(-1,3,-1,-1),v[:,i:i+1].expand(-1,3,-1,-1),dropout_p=0.) for i in range(4)],dim=1)
        group_error=(via_repeat-per_group).abs().max().item();assert group_error<1e-5
    payload={'scope':'CPU tiny numerical equivalence; no GPU timing or deployed cache',
       'torch_version':torch.__version__,'seed':5962,'input_tokens':8,'prefill_tokens':4,
       'single_token_decode_calls':4,'cached_vs_full_max_abs_error':error,
       'cache_total_bytes_float32':cache_bytes,'per_layer_cache_shape':list(caches[0][0].shape),
       'sdpa_cpu_vs_reference_max_abs_error':sdpa_error,'gqa_group_mapping_max_abs_error':group_error,
       'gqa_note':'repeat/expand checks semantics, not a memory-efficient kernel'}
    s=json.dumps(payload,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(s+'\n')
    print(s)
if __name__=='__main__':main()
