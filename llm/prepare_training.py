"""
Prepare full training data for v2 model:
1. Use ALL 52K Alpaca examples (not just 100)
2. Convert to clean v2 format (plain prompt + completion, no embedded tags)
3. Train BPE tokenizer on the full corpus
4. Save dataset + tokenizer for training
"""
import json
import os
import random
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPACA_PATH = "alpaca_raw.json"
TOOL_DATA_PATH = "dataset.json"
OUTPUT_DATASET = "dataset_v2_full.json"
TOKENIZER_OUTPUT = "bpe_tokenizer.json"

def convert_alpaca_to_v2(alpaca_data, limit=None):
    """Convert Alpaca format to clean v2 format (no embedded tags)."""
    converted = []
    items = alpaca_data if limit is None else alpaca_data[:limit]
    for item in items:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction
        converted.append({"prompt": prompt, "completion": output})
    return converted

def convert_tool_data_to_v2(tool_data):
    """Convert AIOS tool data to clean v2 format."""
    converted = []
    for item in tool_data:
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        # Strip any existing tags if present
        for tag in ["<system>", "<user>", "<assistant>", "<eos>"]:
            prompt = prompt.replace(tag, "").strip()
            completion = completion.replace(tag, "").strip()
        if not completion:
            continue
        converted.append({"prompt": prompt, "completion": completion})
    return converted

def main():
    # Load Alpaca data
    if not os.path.exists(ALPACA_PATH):
        print(f"Downloading Stanford Alpaca dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        with urllib.request.urlopen(url) as resp:
            alpaca_raw = json.loads(resp.read().decode("utf-8"))
        with open(ALPACA_PATH, "w", encoding="utf-8") as f:
            json.dump(alpaca_raw, f, ensure_ascii=False)
        print(f"Downloaded {len(alpaca_raw)} examples")
    else:
        with open(ALPACA_PATH, "r", encoding="utf-8") as f:
            alpaca_raw = json.load(f)
        print(f"Loaded {len(alpaca_raw)} Alpaca examples from cache")

    # Convert ALL Alpaca examples
    alpaca_v2 = convert_alpaca_to_v2(alpaca_raw)
    print(f"Converted {len(alpaca_v2)} Alpaca examples to v2 format")

    # Load and convert tool data
    if os.path.exists(TOOL_DATA_PATH):
        with open(TOOL_DATA_PATH, "r", encoding="utf-8") as f:
            tool_raw = json.load(f)
        tool_v2 = convert_tool_data_to_v2(tool_raw)
        print(f"Loaded {len(tool_v2)} tool examples")
    else:
        tool_v2 = []
        print("No tool data found")

    # Combine: all Alpaca + repeated tool data for balance
    repeat_factor = max(1, len(alpaca_v2) // max(1, len(tool_v2)) // 10)
    combined = alpaca_v2 + tool_v2 * repeat_factor
    random.shuffle(combined)
    print(f"Combined dataset: {len(combined)} examples (Alpaca: {len(alpaca_v2)}, Tool repeated {repeat_factor}x)")

    # Save dataset
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset to {OUTPUT_DATASET}")

    # Train BPE tokenizer on all text
    print(f"\nTraining BPE tokenizer on {len(combined)} samples...")
    from tokenizer_v2 import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=32768)

    # Collect text samples for tokenizer training
    texts = []
    for item in combined:
        texts.append(f"<user>{item['prompt']}<assistant>{item['completion']}")
    # Also add system prompts
    for _ in range(100):
        texts.append("<system>You are AIOS v2, a helpful AI assistant.<user>Hello<assistant>Hi! How can I help you?")

    tokenizer.train(texts, verbose=False)

    # Save tokenizer
    save_data = {
        "vocab_size": tokenizer.vocab_size,
        "vocab": tokenizer.vocab,
        "merges": {f"{k[0]},{k[1]}": v for k, v in tokenizer.merges.items()},
        "special_tokens": tokenizer.special_tokens,
        "byte_encoder": {str(k): v for k, v in tokenizer.byte_encoder.items()},
    }
    with open(TOKENIZER_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Saved tokenizer to {TOKENIZER_OUTPUT} (vocab size: {len(tokenizer.vocab)})")

    # Test
    test = tokenizer.encode("<user>What is AI?<assistant>AI is artificial intelligence.")
    decoded = tokenizer.decode(test)
    print(f"Tokenizer test: {len(test)} tokens -> '{decoded[:80]}...'")

    print("\nDone! Ready for training.")

if __name__ == "__main__":
    main()
