"""Small actual CPU checks for Ch37 and Ch39; no 195M or multi-GPU execution."""
import json, math
from pathlib import Path
import torch
from book_transformer import BookTransformer, ModelConfig, tiny_config

def run():
    torch.set_num_threads(1); torch.manual_seed(3701)
    cfg=ModelConfig(vocab_size=16,hidden_size=32,num_layers=2,num_heads=4,intermediate_size=64,max_seq_len=16,attention_backend='sdpa')
    model=BookTransformer(cfg); opt=torch.optim.AdamW(model.parameters(),lr=0.008,weight_decay=0.0)
    keys=torch.arange(1,5)
    inputs=torch.full((4,6),9,dtype=torch.long); inputs[:,0]=keys; inputs[:,-1]=10
    labels=torch.full_like(inputs,-100); labels[:,-1]=keys
    initial=float(model(inputs,labels)['loss'].detach())
    for step in range(100):
        opt.zero_grad(set_to_none=True); model(inputs,labels)['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(inputs,labels); preds=out['logits'][:,-1].argmax(-1); final=float(out['loss'])
        # Every last input token is the same, yet predictions depend on the distant key.
        changed=inputs.clone(); changed[:,0]=keys.roll(1)
        changed_preds=model(changed)['logits'][:,-1].argmax(-1)
        prefix_only=model(inputs[:,:1])['logits'][:,-1].argmax(-1)
    assert torch.equal(preds,keys) and torch.equal(changed_preds,keys.roll(1))
    assert final<0.02
    # Dtype audit follows the reference implementation's actual residual and parameter choices.
    audit_model=BookTransformer(tiny_config()); records=[]; handles=[]
    names=['embedding','blocks.0.attn_norm','blocks.0.attn.q_proj','blocks.0.attn.out_proj',
           'blocks.0.ffn_norm','blocks.0.ffn.gate_proj','blocks.0.ffn.down_proj','blocks.0','final_norm','lm_head']
    for name,module in audit_model.named_modules():
        if name in names:
            def hook(mod,args,out,name=name):
                records.append({'module':name,'input_dtype':str(args[0].dtype),'output_dtype':str(out.dtype),'shape':list(out.shape)})
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16): audit_model(torch.tensor([[1,2,3,4]]))
    for handle in handles: handle.remove()
    result={'device':'cpu','torch_version':torch.__version__,'delayed_copy':{'parameters':sum(p.numel() for p in model.parameters()),
            'updates':100,'supervised_tokens':400,'initial_loss':initial,'final_loss':final,
            'targets':keys.tolist(),'predictions':preds.tolist(),'changed_key_predictions':changed_preds.tolist(),
            'last_token_only_optimal_loss_for_balanced_keys':math.log(4),
            'scope':'fixed six-position diagnostic, no held-out length or natural-language generalization claim'},
            'cpu_bfloat16_module_dtypes':records,'parameter_dtype':str(next(audit_model.parameters()).dtype),
            'gpu_amp_executed':False,'ddp_executed':False,'formal_195m_training_executed':False}
    print(json.dumps(result,ensure_ascii=False,indent=2)); return result
if __name__=='__main__': run()
