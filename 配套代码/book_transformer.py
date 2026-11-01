"""Book reference decoder. Requires PyTorch 2.5+; no pretrained weights.
forward(input_ids, labels=None, loss_mask=None, document_ids=None) returns a dict.
Run `python verify_transformer.py` for CPU tiny checks and meta-only 195M count.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math, os, random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 768
    num_layers: int = 24
    num_heads: int = 12
    intermediate_size: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    attention_backend: str = 'reference'
    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        assert (self.hidden_size // self.num_heads) % 2 == 0
        assert min(self.vocab_size, self.num_layers, self.max_seq_len) > 0
        assert self.attention_backend in ('reference', 'sdpa')

def tiny_config():
    return ModelConfig(vocab_size=32, hidden_size=32, num_layers=2,
                       num_heads=4, intermediate_size=64, max_seq_len=32)

class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(width)); self.eps = eps
    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

def apply_rope(x, theta=10000.0):
    # x: [B,h,T,head_dim]. Adjacent pairs, float32 angles, zero-based positions.
    t, dh = x.shape[-2:]
    inv = theta ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)

def attention_reference(q, k, v, allowed):
    # .float() alone does not disable autocast for eligible matrix multiplies.
    with torch.autocast(q.device.type, enabled=False):
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~allowed, float('-inf')).softmax(dim=-1)
    return weights.to(v.dtype) @ v

class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.h = cfg.num_heads; self.dh = cfg.hidden_size // self.h; self.theta = cfg.rope_theta
        self.backend = cfg.attention_backend
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
    def forward(self, x, allowed):
        b, t, d = x.shape
        def split(p): return p(x).view(b, t, self.h, self.dh).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)
        if self.backend == 'sdpa':
            # PyTorch chooses the supported backend; GPU Flash kernels are not guaranteed.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        else:
            y = attention_reference(q, k, v, allowed)
        return self.out_proj(y.transpose(1, 2).contiguous().view(b, t, d))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = CausalAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)
    def forward(self, x, allowed):
        x = x + self.attn(self.attn_norm(x), allowed)
        return x + self.ffn(self.ffn_norm(x))

class BookTransformer(nn.Module):
    def __init__(self, cfg=None):
        super().__init__(); self.config = cfg or ModelConfig(); cfg = self.config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.gradient_checkpointing = False
        # Initialize each unique weight once; the tied head is not initialized again.
        for name, p in self.named_parameters():
            if name.endswith('norm.weight'): nn.init.ones_(p)
            else: nn.init.normal_(p, mean=0.0, std=0.02)
    def forward(self, input_ids, labels=None, loss_mask=None, document_ids=None):
        if input_ids.ndim != 2: raise ValueError('input_ids must be [B,T]')
        b, t = input_ids.shape
        if t > self.config.max_seq_len: raise ValueError('sequence exceeds configured context')
        allowed = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()[None, None]
        if document_ids is not None:
            if document_ids.shape != input_ids.shape: raise ValueError('document_ids shape mismatch')
            same = document_ids[:, :, None] == document_ids[:, None, :]
            allowed = allowed & same[:, None]
        x = self.embedding(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, allowed, use_reentrant=False)
            else: x = block(x, allowed)
        logits = self.lm_head(self.final_norm(x))
        result = {'logits': logits, 'loss': None, 'loss_sum': None, 'token_count': None}
        if labels is not None:
            if labels.shape != input_ids.shape: raise ValueError('labels must already be shifted to [B,T]')
            per_token = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1),
                                        ignore_index=-100, reduction='none').view(b,t)
            valid = labels.ne(-100).to(per_token.dtype)
            if loss_mask is not None:
                if loss_mask.shape != labels.shape: raise ValueError('loss_mask shape mismatch')
                if not torch.all((loss_mask == 0) | (loss_mask == 1)): raise ValueError('loss_mask must be binary')
                valid = valid * loss_mask
            count = valid.sum(); total = (per_token * valid).sum()
            result.update(loss=total / count.clamp_min(1), loss_sum=total, token_count=count)
        return result

def save_checkpoint(path, model, optimizer, step, tokens, sampler_state=None, *, scheduler=None, scaler=None, metadata=None):
    """Only call after optimizer.step and zero_grad; load only trusted checkpoints."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=1, config=asdict(model.config), model=model.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, tokens=tokens,
                 sampler_state=sampler_state, torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    state['metadata'] = metadata
    if scheduler is not None: state['scheduler'] = scheduler.state_dict()
    if scaler is not None: state['scaler'] = scaler.state_dict()
    if torch.cuda.is_available(): state['cuda_rng'] = torch.cuda.get_rng_state_all()
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def load_checkpoint(path, model, optimizer, *, scheduler=None, scaler=None, expected_metadata=None):
    state = torch.load(path, map_location='cpu', weights_only=False)  # trusted local artifacts only
    if state['format_version'] != 1: raise ValueError('unsupported checkpoint format')
    if state['config'] != asdict(model.config): raise ValueError('configuration mismatch')
    if expected_metadata is not None and state.get('metadata') != expected_metadata: raise ValueError('data or training plan mismatch')
    model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(state['scheduler'])
    if scaler is not None: scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
    if 'cuda_rng' in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
