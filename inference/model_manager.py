import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal, Union
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    TextStreamer,
)

from training.utils import get_logger, get_gpu_memory

logger = get_logger(__name__)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gemma": {
        "models": ["google/gemma-2b", "google/gemma-7b", "google/gemma-2-2b"],
        "trust_remote_code": False,
    },
    "qwen": {
        "models": [
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-1.5B",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2-72B",
        ],
        "trust_remote_code": True,
    },
    "llama": {
        "models": [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-3-8b",
            "NousResearch/Llama-2-7b-hf",
        ],
        "trust_remote_code": False,
    },
    "mistral": {
        "models": [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-v0.1",
        ],
        "trust_remote_code": False,
    },
    "phi": {
        "models": [
            "microsoft/phi-1_5",
            "microsoft/phi-2",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
        "trust_remote_code": True,
    },
    "deepseek": {
        "models": [
            "deepseek-ai/deepseek-llm-7b-base",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ],
        "trust_remote_code": True,
    },
}


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop_strings: Optional[List[str]] = None
    stream: bool = False


@dataclass
class InferenceResult:
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelManager:
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        trust_remote_code: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_flash_attention = use_flash_attention

        self.family = self._detect_family()

        if trust_remote_code is None:
            trust_remote_code = self._get_trust_remote_code()
        self.trust_remote_code = trust_remote_code

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if torch_dtype is None:
            if self.device.type == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        self._loaded = False

    def _detect_family(self) -> str:
        name_lower = self.model_name.lower()
        for family, info in MODEL_REGISTRY.items():
            for m in info["models"]:
                if m.lower() in name_lower or name_lower in m.lower():
                    return family
        return "unknown"

    def _get_trust_remote_code(self) -> bool:
        for info in MODEL_REGISTRY.values():
            if self.model_name in info["models"]:
                return info["trust_remote_code"]
        return False

    def load(self) -> None:
        if self._loaded:
            logger.info("Model already loaded: %s", self.model_name)
            return

        logger.info(
            "Loading model '%s' (family=%s, device=%s, dtype=%s)",
            self.model_name, self.family, self.device, self.torch_dtype,
        )

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_implementation = "flash_attention_2" if self.use_flash_attention else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=attn_implementation,
        )

        if self.device.type != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model loaded: %s (%.2fB params)",
            self.model_name, n_params / 1e9,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded: %s", self.model_name)

    def switch_model(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        self.unload()
        self.model_name = model_name
        self.family = self._detect_family()
        self.trust_remote_code = self._get_trust_remote_code()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.load()

    @torch.no_grad()
    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        start = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        tokens_generated = len(generated)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if request.stop_strings:
            for stop_str in request.stop_strings:
                if stop_str in text:
                    text = text[: text.index(stop_str)]

        result = InferenceResult(
            text=text,
            tokens_generated=tokens_generated,
            latency_ms=elapsed * 1000,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            model_name=self.model_name,
        )

        return result

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        batch_size: int = 4,
        **kwargs,
    ) -> List[InferenceResult]:
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for prompt in batch:
                request = InferenceRequest(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                result = self.generate(request)
                results.append(result)
        return results

    def get_chat_template(self) -> Optional[str]:
        if self.tokenizer and self.tokenizer.chat_template:
            return self.tokenizer.chat_template
        return None

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> InferenceResult:
        prompt = self.apply_chat_template(messages)
        request = InferenceRequest(prompt=prompt, **kwargs)
        return self.generate(request)

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_name": self.model_name,
            "family": self.family,
            "loaded": self._loaded,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "trust_remote_code": self.trust_remote_code,
        }

        if self._loaded and self.model is not None:
            n_params = sum(p.numel() for p in self.model.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.update({
                "parameters_b": round(n_params / 1e9, 2),
                "trainable_b": round(n_trainable / 1e9, 2),
            })

        if self.device.type == "cuda":
            mem = get_gpu_memory()
            if mem:
                info["gpu_memory_mb"] = round(mem, 1)

        return info
