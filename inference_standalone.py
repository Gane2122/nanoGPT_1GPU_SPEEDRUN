#!/usr/bin/env python3
"""
Standalone inference script for trained GPT-2 model
Extracted from train_gpt_optimized.py without training dependencies
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tiktoken

os.environ["DISABLE_FP8"] = "1"  # Disable FP8 for inference
device = "cuda"

# -----------------------------------------------------------------------------
# Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    """Standard Rotary Position Embeddings - matches train_gpt_improved.py"""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        # Separate Q/K/V/O projections - matches train_gpt_improved.py
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.c_proj = CastedLinear(dim, dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))  # Value residual lambda
        self.rotary = Rotary(dim // num_heads)
        self.attn_gate = CastedLinear(12, num_heads)

    def forward(self, x: Tensor, v1=None, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn_scale)
        y = y.transpose(1, 2)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, v1, x0: Tensor, attn_scale=0.1):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # Match train_gpt_improved.py structure
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, model_dim),
            h=nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)]),
        ))
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))
        self.num_layers = num_layers

    def forward(self, input_seq: Tensor):
        """Inference-only forward pass"""
        if input_seq.ndim == 1:
            input_seq = input_seq.unsqueeze(0)
        B, T = input_seq.shape

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]

        x = self.transformer.wte(input_seq)

        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
        x = norm(x)
        x0 = x
        v1 = None

        skip_connections = []
        x_backout = None
        backout_layer = 8

        # Encoder pass
        for i in range(self.num_encoder_layers):
            if ve_list[i] is not None:
                if v1 is None:
                    v1 = ve_list[i][None].view(B, T, self.transformer.h[i].attn.num_heads if self.transformer.h[i].attn else 6, -1)
            x, v1 = self.transformer.h[i](x, v1, x0, attn_scale=0.1)
            skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # Decoder pass with weighted skip connections
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None:
                if v1 is None:
                    v1 = ve_list[layer_idx][None].view(B, T, 6, -1)
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, attn_scale=0.1)
            if layer_idx == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)
        # Use tanh logit scaling like train_gpt_improved.py
        logits = 30 * torch.tanh(logits / 30)
        return logits

# -----------------------------------------------------------------------------
# Text Generation

@torch.no_grad()
def generate(model, enc, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
    """Generate text from prompt"""
    BOS_ID = 50256  # GPT-2 BOS/EOS token
    tokens = [BOS_ID] + enc.encode(prompt)  # Prepend BOS token like training
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == enc.eot_token or tokens.size(1) > 1024:
            break

    # Strip BOS token from output
    output_tokens = tokens[0].tolist()[1:]  # Remove BOS at index 0
    return enc.decode(output_tokens)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    checkpoint_path = "gpt2-124m-fromscratch-step1750-3.2860 loss.pt"
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint loaded. Step: {checkpoint.get('step', 'N/A')}")

    # Create model with same config
    model = GPT(
        vocab_size=50257,
        num_layers=12,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=16384
    )

    # Load weights - handle _orig_mod. prefix from torch.compile
    state_dict = {}
    for k, v in checkpoint['model'].items():
        new_k = k.replace("_orig_mod.", "")
        state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    model = model.to(device).eval()
    print(f"Model loaded successfully")

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer loaded")

    # Test prompts
    prompts = [
        "The capital of France is",
        "In the field of machine learning,",
        "Once upon a time, there was a",
        "The most important scientific discovery of the 20th century was",
    ]

    print("\n" + "="*60)
    print("INFERENCE TEST RESULTS")
    print("="*60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")

        for temp in [0.7, 1.0]:
            output = generate(model, enc, prompt, max_new_tokens=50, temperature=temp, top_k=40)
            print(f"\nTemperature {temp}:")
            print(output)

        print("\n" + "-"*60)

    print("\nInference test complete!")
