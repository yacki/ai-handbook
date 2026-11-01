"""CPU toy adaptation using the book's shared decoder; requires PyTorch.
Synthetic token IDs only, not a pretrained 195M model or instruction benchmark.
Run: python ch47_48_tiny_adaptation.py --output ../微型适配实验记录.json
"""
import argparse, json, math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from book_transformer import BookTransformer, tiny_config

class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__(); self.base = base; self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale
    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)

def batch():
    x = torch.tensor([[1,3,q,4,a,a+4] for q,a in zip(range(5,9),range(10,14))])
    y = torch.cat((x[:,1:], torch.full((4,1),2)),dim=1)
    mask = torch.tensor([[0,0,0,1,1,1]]*4)
    return x,y,mask

def run(mode):
    torch.manual_seed(470)
    model = BookTransformer(tiny_config())
    if mode == 'lora':
        model.requires_grad_(False)
        for block in model.blocks:
            block.attn.q_proj = LoRALinear(block.attn.q_proj)
            block.attn.v_proj = LoRALinear(block.attn.v_proj)
    original_base = {n: p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable,lr=.015 if mode=='sft' else .05,weight_decay=0.)
    x,y,mask = batch()
    initial = model(x,y,mask)['loss'].item()
    steps = 100 if mode=='sft' else 150
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x,y,mask); out['loss'].backward(); opt.step()
    with torch.no_grad():
        final = model(x,y,mask)['loss'].item()
        argmax = model(x)['logits'].argmax(-1)
        correct = ((argmax == y) & mask.bool()).sum().item()
        base_max_change = max((model.get_parameter(n)-p).abs().max().item() for n,p in original_base.items()) if original_base else None
        merge_error = None
        if mode=='lora':
            layer = model.blocks[0].attn.q_proj
            test_input = torch.randn(2,3,32)
            merge_error = (layer(test_input)-F.linear(test_input,layer.merged_weight())).abs().max().item()
    assert final < initial
    if mode=='lora':
        assert base_max_change == 0.
        assert merge_error < 1e-5
    return {'mode':mode,'seed':470,'steps':steps,'input_shape':list(x.shape),'supervised_targets':int(mask.sum()),
            'trainable_parameters':sum(p.numel() for p in trainable),'initial_loss':initial,'final_loss':final,
            'teacher_forced_correct_targets':correct,'frozen_base_max_change':base_max_change,
            'merge_max_abs_error':merge_error,'note':'toy memorization and numerical checks, not instruction capability'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    torch.set_num_threads(1)
    payload={'torch_version':torch.__version__,'device':'cpu','config':vars(tiny_config()),'runs':[run('sft'),run('lora')]}
    result=json.dumps(payload,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(result+'\n')
    print(result)
if __name__=='__main__':main()
