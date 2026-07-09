"""
Download and prepare real instruction-following data from Stanford Alpaca.
Merges 3000 Alpaca examples with our 76 AIOS tool-calling examples
to create a combined dataset for training.
"""
import urllib.request
import json
import os
import random

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
ALPACA_CACHE = "alpaca_raw.json"
OUTPUT_FILE = "dataset_full.json"
ALPACA_LIMIT = 100  # Use first 100 for fast demo training

def download_alpaca():
    if os.path.exists(ALPACA_CACHE):
        print(f"Using cached Alpaca data: {ALPACA_CACHE}")
        with open(ALPACA_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"Downloading Stanford Alpaca dataset...")
    with urllib.request.urlopen(ALPACA_URL) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    with open(ALPACA_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    print(f"Downloaded {len(data)} examples. Cached to {ALPACA_CACHE}")
    return data

def convert_alpaca_to_aios_format(alpaca_data, limit=ALPACA_LIMIT):
    """Convert Alpaca format to our AIOS prompt/completion format."""
    converted = []
    for item in alpaca_data[:limit]:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        if not instruction or not output:
            continue
        
        # Format: if input exists, include it in user message
        if input_text:
            user_msg = f"{instruction}\n{input_text}"
        else:
            user_msg = instruction
        
        converted.append({
            "prompt": f"<system>You are AIOS, a highly intelligent AI agent.<user>{user_msg}<assistant>",
            "completion": f"{output}<eos>"
        })
    
    return converted

def load_tool_dataset():
    """Load our AIOS tool-calling dataset."""
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Download Alpaca
    alpaca_raw = download_alpaca()
    
    # Convert to our format
    alpaca_examples = convert_alpaca_to_aios_format(alpaca_raw)
    print(f"Converted {len(alpaca_examples)} Alpaca examples")
    
    # Load AIOS tool examples
    tool_examples = load_tool_dataset()
    print(f"Loaded {len(tool_examples)} AIOS tool examples")
    
    # Merge and shuffle
    combined = alpaca_examples + tool_examples * 10  # Repeat tool examples 10x for balance
    random.shuffle(combined)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    print(f"Combined dataset: {len(combined)} examples -> saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
