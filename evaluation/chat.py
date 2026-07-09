from typing import List, Optional, Dict, Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluate import ModelEvaluator, EvaluationResult
from training.utils import get_logger

logger = get_logger(__name__)


CHAT_TEMPLATES = [
    [
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    [
        {"role": "user", "content": "Write a short poem about artificial intelligence."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 tips for learning Python."},
    ],
    [
        {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"},
    ],
    [
        {"role": "user", "content": "Translate 'Hello, how are you?' to French, Spanish, and German."},
    ],
]


class ChatEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.evaluator = ModelEvaluator(model, tokenizer)

    def evaluate(
        self,
        conversations: Optional[List[List[Dict[str, str]]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        batch_size: int = 1,
    ) -> EvaluationResult:
        convos = conversations or CHAT_TEMPLATES
        prompts = []
        for convo in convos:
            text = self.evaluator.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

        logger.info("Evaluating chat on %d conversations", len(prompts))

        return self.evaluator.evaluate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            batch_size=batch_size,
        )
