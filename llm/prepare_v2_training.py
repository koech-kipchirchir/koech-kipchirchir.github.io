"""
Prepare training data for v2 model using fixed BPE tokenizer.
1. Load ALL 52K Alpaca examples + tool data
2. Train BPE tokenizer on full corpus
3. Save dataset + tokenizer for training
"""
import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPACA_PATH = "alpaca_raw.json"
TOOL_DATA_PATH = "dataset.json"
OUTPUT_DATASET = "dataset_v2_full.json"
TOKENIZER_OUTPUT = "bpe_tokenizer.json"

def convert_alpaca(alpaca_data):
    converted = []
    for item in alpaca_data:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue
        prompt = f"{instruction}\n{input_text}" if input_text else instruction
        converted.append({"prompt": prompt, "completion": output})
    return converted

def convert_tool_data(tool_data):
    converted = []
    for item in tool_data:
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        for tag in ["<system>", "<user>", "<assistant>", "<eos>"]:
            prompt = prompt.replace(tag, "").strip()
            completion = completion.replace(tag, "").strip()
        if completion:
            converted.append({"prompt": prompt, "completion": completion})
    return converted

def main():
    # Load Alpaca
    if not os.path.exists(ALPACA_PATH):
        print("Downloading Stanford Alpaca dataset...")
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
        print(f"Loaded {len(alpaca_raw)} Alpaca examples")

    alpaca_v2 = convert_alpaca(alpaca_raw)
    print(f"Converted {len(alpaca_v2)} Alpaca examples")

    # Load tool data
    tool_v2 = []
    if os.path.exists(TOOL_DATA_PATH):
        with open(TOOL_DATA_PATH, "r", encoding="utf-8") as f:
            tool_raw = json.load(f)
        tool_v2 = convert_tool_data(tool_raw)
        print(f"Loaded {len(tool_v2)} tool examples")

    # Combine
    repeat = max(1, len(alpaca_v2) // max(1, len(tool_v2)) // 50)
    combined = alpaca_v2 + tool_v2 * repeat
    random.shuffle(combined)
    print(f"Combined: {len(combined)} examples ({len(alpaca_v2)} Alpaca, tools x{repeat})")

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset to {OUTPUT_DATASET}")

    # Train BPE tokenizer
    print(f"\nTraining BPE tokenizer on {len(combined)} samples...")
    from bpe_tokenizer_fixed import BPETokenizerFixed

    tokenizer = BPETokenizerFixed(vocab_size=32768)

    # Collect training texts
    texts = []
    for item in combined[:20000]:  # Use subset for faster tokenizer training
        texts.append(f"<user>{item['prompt']}<assistant>{item['completion']}")
    # Add system prompts
    for i in range(min(500, len(combined))):
        item = combined[i]
        texts.append(f"<system>You are AIOS v2, a helpful AI assistant.<user>{item['prompt']}<assistant>{item['completion']}")

    random.shuffle(texts)
    tokenizer.train(texts, verbose=True)

    # Save tokenizer via wrapper
    from bpe_tokenizer_fixed import BPETokenizerWrapper
    wrapper = BPETokenizerWrapper(vocab_size=32768)
    wrapper.bpe = tokenizer
    wrapper.save(TOKENIZER_OUTPUT)
    print(f"Saved tokenizer to {TOKENIZER_OUTPUT}")

    # Test
    test = "What is artificial intelligence?"
    enc = tokenizer.encode(test)
    dec = tokenizer.decode(enc)
    print(f"\nTokenizer test: '{test}' -> {len(enc)} tokens -> '{dec}'")
    print(f"Vocab size: {len(tokenizer.vocab)}")

    print("\nDone! Ready for training.")

if __name__ == "__main__":
    main()
