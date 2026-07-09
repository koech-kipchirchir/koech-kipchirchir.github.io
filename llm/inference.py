"""
AIOS-LLM Inference Engine
Loads the trained checkpoint (original GPT-style architecture) and generates completions.
Auto-detects checkpoint architecture from saved keys.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import CharacterTokenizer


# ── Minimal original GPT model (matches the trained checkpoint keys) ────────
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class OriginalGPTConfig:
    vocab_size: int = 100
    block_size: int = 256
    n_layer:    int = 3
    n_head:     int = 4
    n_embd:     int = 128
    dropout:    float = 0.05


class OriginalCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.register_buffer("bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class OriginalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class OriginalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = OriginalCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = OriginalMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OriginalGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([OriginalBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9, eos_token_id=2):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
            if next_tok.item() == eos_token_id:
                break
        return idx


# ── Inference Engine ─────────────────────────────────────────────────────────
class AIOSInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Rebuild tokenizer
        self.tokenizer = CharacterTokenizer()
        if "vocab" in ckpt:
            self.tokenizer.vocab = ckpt["vocab"]
            self.tokenizer.token_to_id = {t: i for i, t in enumerate(self.tokenizer.vocab)}
            self.tokenizer.id_to_token = {i: t for i, t in enumerate(self.tokenizer.vocab)}

        # Detect architecture from checkpoint keys
        keys = list(ckpt["model_state_dict"].keys())
        use_original = any("transformer.wte" in k for k in keys)

        if use_original:
            cfg_saved = ckpt.get("config")
            if cfg_saved and hasattr(cfg_saved, "n_layer"):
                cfg = OriginalGPTConfig(
                    vocab_size = cfg_saved.vocab_size,
                    block_size = cfg_saved.block_size,
                    n_layer    = cfg_saved.n_layer,
                    n_head     = cfg_saved.n_head,
                    n_embd     = cfg_saved.n_embd,
                )
            else:
                cfg = OriginalGPTConfig(vocab_size=len(self.tokenizer.vocab))
            self.model = OriginalGPT(cfg).to(self.device)
        else:
            from model import GPT, GPTConfig
            cfg_saved = ckpt.get("config")
            cfg = cfg_saved if cfg_saved else GPTConfig(vocab_size=len(self.tokenizer.vocab))
            self.model = GPT(cfg).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        arch = "Original GPT" if use_original else "LLaMA-style"
        print(f"[AIOS-LLM] Loaded {arch} | {self.model.get_num_params()/1e6:.2f}M params | device={self.device}")

    def generate(self, user_input: str, max_new_tokens: int = 150,
                 temperature: float = 0.75, top_k: int = 30) -> str:
        prompt = f"<system>You are AIOS, a device agent.<user>{user_input.strip()}<assistant>"
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)

        block_size = self.model.config.block_size
        if len(encoded) > block_size - max_new_tokens:
            encoded = encoded[-(block_size - max_new_tokens):]

        idx = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(idx, max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_k=top_k)

        new_tokens = out[0][len(encoded):].tolist()
        raw = self.tokenizer.decode(new_tokens)
        for tok in getattr(self.tokenizer, 'special_tokens', []):
            raw = raw.replace(tok, "")
        return raw.strip()


if __name__ == "__main__":
    engine = AIOSInferenceEngine("aios_llm.pth")
    tests  = ["hello", "who are you?", "turn on the flashlight",
              "set an alarm for 7:30 AM", "vibrate the phone"]
    print("\n" + "="*55 + "\nAIOS-LLM Inference Test\n" + "="*55)
    for p in tests:
        print(f"\n> {p}")
        print(f"  {engine.generate(p)}")
