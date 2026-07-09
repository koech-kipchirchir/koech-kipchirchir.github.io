"""
AIOS-LLM Training Pipeline
- Trains on combined Alpaca + AIOS tool dataset
- Gradient accumulation for larger effective batch sizes
- Cosine LR schedule with linear warmup
- Checkpoint saving every 25 epochs
"""
import json
import os
import sys
import math
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharacterTokenizer
from model import GPT, GPTConfig


# ─── Config ──────────────────────────────────────────────────────────────────
EPOCHS          = 5
BATCH_SIZE      = 8
GRAD_ACCUM      = 4          # effective batch = 8 * 4 = 32
MAX_SEQ_LEN     = 512
LR_MAX          = 5e-4
LR_MIN          = 5e-6
WARMUP_EPOCHS   = 10
CHECKPOINT_DIR  = "checkpoints"
FINAL_MODEL     = "aios_llm.pth"
DATASET_FILE    = "dataset_full.json"  # Use full dataset if available, else fallback

# Model size: ~10M params
MODEL_CONFIG = dict(
    block_size = MAX_SEQ_LEN,
    n_layer    = 4,
    n_head     = 4,
    n_embd     = 128,
    dropout    = 0.05,
)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_SEQ_LEN):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        skipped = 0
        for item in raw:
            prompt     = item.get("prompt", "")
            completion = item.get("completion", "")

            p_ids = tokenizer.encode(prompt,     add_special_tokens=False)
            c_ids = tokenizer.encode(completion, add_special_tokens=False)
            full  = p_ids + c_ids

            if len(full) < 4:
                skipped += 1
                continue

            if len(full) > max_length:
                full = full[:max_length]

            x = full[:-1]
            y = [-1] * (len(p_ids) - 1) + full[len(p_ids):]  # mask prompt in loss

            # Pad
            pad = max_length - len(x)
            x = x + [0] * pad
            y = y + [-1] * pad

            self.samples.append((
                torch.tensor(x[:max_length], dtype=torch.long),
                torch.tensor(y[:max_length], dtype=torch.long),
            ))

        print(f"Dataset: {len(self.samples)} samples loaded ({skipped} skipped)")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ─── LR Schedule ─────────────────────────────────────────────────────────────
def get_lr(epoch, warmup=WARMUP_EPOCHS, total=EPOCHS, lr_max=LR_MAX, lr_min=LR_MIN):
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    tokenizer = CharacterTokenizer()

    # Prefer full dataset, fallback to tool-only dataset
    ds_file = DATASET_FILE if os.path.exists(DATASET_FILE) else "dataset.json"
    print(f"Using dataset: {ds_file}")

    dataset    = InstructionDataset(ds_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    config = GPTConfig(vocab_size=tokenizer.vocab_size, **MODEL_CONFIG)
    model  = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float("inf")

    print(f"\nStarting training: {EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        # Set LR
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss

        log_interval = 25 if len(dataset) > 500 else 10
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Best: {best_loss:.4f}")
            sys.stdout.flush()

        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": config,
                "vocab": tokenizer.vocab,
                "loss": avg_loss,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1}.pth"))

    # Save final model
    final = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab": tokenizer.vocab,
    }
    torch.save(final, FINAL_MODEL)
    print(f"\nAIOS-LLM training complete! Final loss: {best_loss:.4f}")
    print(f"Saved: {FINAL_MODEL}")


if __name__ == "__main__":
    train()
