"""
AIOS v2 Training - Optimized for CPU training with byte-level tokenizer.
Uses the v2 model architecture with vocab_size=260 for fast training.
"""
import json
import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from byte_tokenizer import ByteTokenizerWrapper
from model_v2 import AIOSModelV2, ModelConfig


# ─── Config ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 260  # 4 special + 256 bytes
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 688
NUM_LAYERS = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 5e-4
MIN_LR = 5e-6
WARMUP_STEPS = 200
TOTAL_STEPS = 20000
SAVE_EVERY = 2000
EVAL_EVERY = 500
OUTPUT_DIR = "checkpoints_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            items = []
            for key in ["train", "data", "samples", "conversations"]:
                if key in raw:
                    items = raw[key]
                    break
        else:
            items = raw

        for item in items:
            if isinstance(item, dict):
                prompt = item.get("prompt") or item.get("instruction") or item.get("input") or ""
                completion = item.get("completion") or item.get("output") or item.get("response") or ""
                if not completion:
                    continue
                text = f"<user>{prompt}<assistant>{completion}"
            elif isinstance(item, str):
                text = item
            else:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 4:
                continue
            self.samples.append(ids)

        print(f"Dataset: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        seq = tokens[:self.max_length]
        x = seq[:-1]
        y = seq[1:]

        pad_len = self.max_length - len(x)
        if pad_len > 0:
            x = x + [0] * pad_len
            y = y + [-1] * pad_len

        return (
            torch.tensor(x[:self.max_length], dtype=torch.long),
            torch.tensor(y[:self.max_length], dtype=torch.long),
        )


# ─── LR Schedule ──────────────────────────────────────────────────────────────
def get_lr(step, warmup=WARMUP_STEPS, total=TOTAL_STEPS, lr_max=LEARNING_RATE, lr_min=MIN_LR):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CPU" if device.type == "cpu" else "GPU"
    print(f"\n{'='*60}")
    print(f"AIOS v2 Training on {device_name}")
    print(f"{'='*60}\n")

    # Tokenizer
    tokenizer = ByteTokenizerWrapper(vocab_size=VOCAB_SIZE)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Dataset
    dataset_path = "dataset_v2_full.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        # Fallback: use existing dataset
        for alt in ["dataset_full.json", "dataset.json"]:
            if os.path.exists(alt):
                dataset_path = alt
                print(f"Using fallback: {dataset_path}")
                break
        else:
            print("No dataset found!")
            return

    dataset = TextDataset(dataset_path, tokenizer, MAX_SEQ_LEN)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_SEQ_LEN,
        dropout=0.05,
    )
    model = AIOSModelV2(config).to(device)
    n_params = model.get_num_params()
    print(f"Model: {n_params/1e6:.2f}M params, {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Training loop
    global_step = 0
    best_loss = float('inf')
    val_loss = float('inf')
    model.train()
    optimizer.zero_grad()

    print(f"\nStarting training: {TOTAL_STEPS} steps, batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"{'='*60}")

    start_time = time.time()
    step_times = []

    microbatch = 0
    while global_step < TOTAL_STEPS:
        for batch in train_loader:
            if global_step >= TOTAL_STEPS:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            logits, loss, _ = model(x, targets=y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            microbatch += 1

            if microbatch % GRAD_ACCUM == 0:
                # Step
                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_times.append(time.time())
                if len(step_times) > 100:
                    step_times.pop(0)

                # Log
                if global_step % 50 == 0:
                    steps_per_sec = len(step_times) / max(1, (step_times[-1] - step_times[0]))
                    elapsed = time.time() - start_time
                    remaining = (TOTAL_STEPS - global_step) / max(0.1, steps_per_sec)
                    print(f"Step {global_step:06d}/{TOTAL_STEPS} | Loss: {loss.item()*GRAD_ACCUM:.4f} | "
                          f"LR: {lr:.2e} | {steps_per_sec:.1f} step/s | "
                          f"ETA: {remaining/60:.0f}m")

                # Eval
                if global_step % EVAL_EVERY == 0:
                    model.eval()
                    val_loss = 0.0
                    num_val = 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            _, vl, _ = model(vx, targets=vy)
                            val_loss += vl.item()
                            num_val += 1
                    val_loss /= max(1, num_val)
                    ppl = math.exp(val_loss)

                    imp = " (BEST!)" if val_loss < best_loss else ""
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "model_config": config,
                            "val_loss": val_loss,
                        }, os.path.join(OUTPUT_DIR, "best.pt"))

                    print(f"  EVAL: loss={val_loss:.4f}, ppl={ppl:.2f}{imp}")
                    model.train()

                # Save checkpoint
                if global_step % SAVE_EVERY == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_step_{global_step}.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_config": config,
                        "global_step": global_step,
                        "val_loss": val_loss if global_step % EVAL_EVERY == 0 else 0,
                    }, ckpt_path)
                    print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(OUTPUT_DIR, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "global_step": global_step,
        "val_loss": val_loss,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    # Also save as the default checkpoint name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
    }, os.path.join(OUTPUT_DIR, "best.pt"))

    elapsed = time.time() - start_time
    print(f"\nTraining complete! {elapsed/60:.1f}m elapsed")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Model saved to {OUTPUT_DIR}/best.pt")


if __name__ == "__main__":
    train()
