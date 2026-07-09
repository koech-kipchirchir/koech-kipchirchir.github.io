import re
from typing import List, Optional, Callable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluate import ModelEvaluator, EvaluationResult
from training.utils import get_logger

logger = get_logger(__name__)


MATH_PROMPTS = [
    "What is 25 * 4 + 18?",
    "Solve for x: 2x + 5 = 13",
    "What is the area of a circle with radius 7?",
    "If a car travels at 60 mph for 2.5 hours, how far does it go?",
    "What is 15% of 200?",
    "Simplify: (12 + 8) / (2 * 5)",
    "What is the square root of 144?",
    "If x = 3 and y = 4, what is x^2 + y^2?",
]

MATH_ANSWERS = [
    "118",
    "4",
    "153.93804002589985",
    "150",
    "30",
    "2.0",
    "12",
    "25",
]


class MathEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.evaluator = ModelEvaluator(model, tokenizer)

    def evaluate(
        self,
        prompts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 1,
    ) -> EvaluationResult:
        test_prompts = prompts or MATH_PROMPTS
        test_answers = answers or MATH_ANSWERS

        logger.info("Evaluating math on %d problems", len(test_prompts))

        return self.evaluator.evaluate(
            prompts=test_prompts,
            ground_truth=test_answers,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            batch_size=batch_size,
            correct_fn=self._math_answer_correct,
        )

    def _math_answer_correct(self, prediction: str, ground_truth: str) -> bool:
        pred_num = self._extract_number(prediction)
        gt_num = self._extract_number(ground_truth)
        if pred_num is None or gt_num is None:
            return False
        return abs(pred_num - gt_num) < 0.01

    def _extract_number(self, text: str) -> Optional[float]:
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        if not numbers:
            return None
        return float(numbers[-1])
