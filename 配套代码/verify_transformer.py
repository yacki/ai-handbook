"""Fast CPU verification; the 195M architecture is constructed on meta only."""
import json, tempfile
from pathlib import Path
import torch
from book_transformer import (BookTransformer, ModelConfig, tiny_config, attention_reference,
                              apply_rope, save_checkpoint, load_checkpoint)

def run():
    torch.set_num_threads(1); torch.manual_seed(20261101)
    report = {'device': 'cpu', 'torch_version': torch.__version__, 'main_model_training': False}
    with torch.device('meta'):
        large = BookTransformer(ModelConfig())
    n = sum(p.numel() for p in large.parameters())
    assert n == 195072768; assert large.embedding.weight is large.lm_head.weight
    report['main_model_meta_parameters'] = n
    model = BookTransformer(tiny_config())
    report['tiny_parameters'] = sum(p.numel() for p in model.parameters())
    x = torch.arange(17)[None].repeat(4,1) % 4 + 1
    ids, labels = x[:, :-1], x[:, 1:]
    model.eval()
    with torch.no_grad():
        out = model(ids)['logits']
        altered = ids.clone(); altered[:, 8:] = 9
        diff = (out[:,:8] - model(altered)['logits'][:,:8]).abs().max().item()
    assert diff < 1e-6; report['future_perturbation_max_diff'] = diff
    # Segment isolation: changing the first document must not affect the second.
    docs = torch.tensor([[0]*8+[1]*8]).repeat(4,1)
    changed = ids.clone(); changed[:,:8] = 12
    with torch.no_grad():
        a = model(ids, document_ids=docs)['logits']
        z = model(changed, document_ids=docs)['logits']
    segdiff = (a[:,8:] - z[:,8:]).abs().max().item()
    assert segdiff < 1e-6; report['document_isolation_max_diff'] = segdiff
    q = torch.randn(2,4,6,8)
    rot = apply_rope(q)
    normdiff = (q.square().sum(-1)-rot.square().sum(-1)).abs().max().item()
    assert normdiff < 1e-5; report['rope_squared_norm_max_diff'] = normdiff
    # Verify reference attention against PyTorch's scaled-dot-product operator.
    k, v = torch.randn_like(q), torch.randn_like(q)
    allowed = torch.ones(6,6,dtype=torch.bool).tril()
    ref = attention_reference(q,k,v,allowed)
    built = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed)
    adiff = (ref-built).abs().max().item(); assert adiff < 1e-5
    report['attention_reference_max_diff'] = adiff
    # Same architecture and weights through the optional PyTorch SDPA backend.
    sdpa_cfg = tiny_config(); sdpa_cfg.attention_backend = 'sdpa'
    sdpa_model = BookTransformer(sdpa_cfg); sdpa_model.load_state_dict(model.state_dict())
    sdiff = (model(ids)['logits'] - sdpa_model(ids)['logits']).abs().max().item()
    assert sdiff < 1e-5; report['sdpa_model_logits_max_diff'] = sdiff
    model.zero_grad(set_to_none=True); sdpa_model.zero_grad(set_to_none=True)
    model(ids,labels)['loss'].backward(); sdpa_model(ids,labels)['loss'].backward()
    sgdiff = max((p.grad-r.grad).abs().max().item() for p,r in zip(model.parameters(), sdpa_model.parameters()))
    assert sgdiff < 1e-5; report['sdpa_model_gradient_max_diff'] = sgdiff
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    initial = model(ids,labels)['loss'].item()
    for step in range(80):
        opt.zero_grad(set_to_none=True); loss = model(ids,labels)['loss']; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    final = model(ids,labels)['loss'].item()
    assert final < 0.08 and final < initial / 10
    report.update(tiny_initial_loss=initial, tiny_final_loss=final, tiny_optimizer_steps=80,
                  tiny_training_tokens=80*ids.numel())
    opt.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'resume.pt'; save_checkpoint(path,model,opt,80,80*ids.numel(),{'next_batch':80})
        restored = BookTransformer(tiny_config()); opt2 = torch.optim.AdamW(restored.parameters(),lr=0.01)
        state = load_checkpoint(path,restored,opt2)
        for m,o in ((model,opt),(restored,opt2)):
            o.zero_grad(set_to_none=True); m(ids,labels)['loss'].backward(); o.step()
        pdiff = max((p-r).abs().max().item() for p,r in zip(model.parameters(),restored.parameters()))
    assert pdiff == 0.0; report['resume_next_update_max_diff'] = pdiff
    # Checkpoint recomputation must give the same gradients without stochastic layers.
    plain = BookTransformer(tiny_config()); remat = BookTransformer(tiny_config()); remat.load_state_dict(plain.state_dict())
    remat.gradient_checkpointing = True
    plain(ids,labels)['loss'].backward(); remat(ids,labels)['loss'].backward()
    gdiff = max((p.grad-r.grad).abs().max().item() for p,r in zip(plain.parameters(),remat.parameters()))
    assert gdiff < 1e-6; report['activation_checkpoint_gradient_max_diff'] = gdiff
    # A masked target must not contribute to the count.
    labels2 = labels.clone(); labels2[:,-2:] = -100
    result = plain(ids,labels2)
    assert result['token_count'].item() == 4*14
    report['masked_token_count'] = int(result['token_count'].item())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__ == '__main__':
    run()
