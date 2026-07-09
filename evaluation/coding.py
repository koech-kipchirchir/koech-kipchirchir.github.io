import ast
import signal
from typing import List, Optional, Callable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluate import ModelEvaluator, EvaluationResult
from training.utils import get_logger

logger = get_logger(__name__)


CODE_PROMPTS = [
    "Write a Python function to check if a string is a palindrome. Include type hints and a docstring.",
    "Write a function that merges two sorted lists into one sorted list.",
    "Implement a LRU cache class in Python with get and put methods.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "Implement a binary search tree class with insert, delete, and search methods.",
    "Write a function to serialize and deserialize a binary tree in Python.",
    "Implement a thread-safe singleton pattern in Python.",
    "Write a Python decorator that measures and prints the execution time of a function.",
]


class CodingEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.evaluator = ModelEvaluator(model, tokenizer)

    def evaluate(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        batch_size: int = 1,
    ) -> EvaluationResult:
        test_prompts = prompts or CODE_PROMPTS
        logger.info("Evaluating coding on %d prompts", len(test_prompts))

        return self.evaluator.evaluate(
            prompts=test_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            batch_size=batch_size,
            correct_fn=self._check_code_syntax,
        )

    def _check_code_syntax(self, prediction: str, _: str) -> bool:
        code = self._extract_code(prediction)
        if not code:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            start = text.find("```python") + 10
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        return text.strip()
