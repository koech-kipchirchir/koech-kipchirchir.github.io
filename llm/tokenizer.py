import json
import os

class CharacterTokenizer:
    def __init__(self):
        # Define special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.system_token = "<system>"
        self.user_token = "<user>"
        self.assistant_token = "<assistant>"
        
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.system_token,
            self.user_token,
            self.assistant_token
        ]
        
        # Base characters to support
        self.chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-+/\\_@:;()[]{}%#\"'=\n\t")
        self.vocab = self.special_tokens + self.chars
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id[self.bos_token])
            
        # Parse text, capturing special tokens vs standard characters
        i = 0
        n = len(text)
        while i < n:
            found_special = False
            for spec in self.special_tokens:
                if text[i:i+len(spec)] == spec:
                    ids.append(self.token_to_id[spec])
                    i += len(spec)
                    found_special = True
                    break
            if not found_special:
                char = text[i]
                if char in self.token_to_id:
                    ids.append(self.token_to_id[char])
                else:
                    # Map unknown characters to space or pad
                    ids.append(self.token_to_id[" "])
                i += 1
                
        if add_special_tokens:
            ids.append(self.token_to_id[self.eos_token])
        return ids
        
    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                tokens.append(token)
        return "".join(tokens)

# Simple self-test
if __name__ == "__main__":
    t = CharacterTokenizer()
    text = "<system>You are AIOS.<user>Vibrate the phone.<assistant>"
    encoded = t.encode(text)
    decoded = t.decode(encoded)
    print(f"Vocab size: {t.vocab_size}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
