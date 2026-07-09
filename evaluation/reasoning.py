from typing import List, Tuple, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluate import ModelEvaluator, EvaluationResult
from training.utils import get_logger

logger = get_logger(__name__)


REASONING_PROMPTS = [
    "If you have a 3-gallon jug and a 5-gallon jug, how can you measure exactly 4 gallons of water?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this logically valid?",
    "You have 12 coins, one of which is counterfeit and either heavier or lighter. Using a balance scale only 3 times, how do you find the counterfeit?",
    "What is the next number in the sequence: 2, 6, 18, 54, ___?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "In a room of 23 people, what is the probability that at least two share a birthday? Explain your reasoning.",
    "A train leaves Station A at 60 mph. Another train leaves Station B at 90 mph. The stations are 300 miles apart. When and where do they meet?",
]


REASONING_GROUND_TRUTHS = [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]


class ReasoningEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.evaluator = ModelEvaluator(model, tokenizer)

    def evaluate(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        batch_size: int = 1,
    ) -> EvaluationResult:
        test_prompts = prompts or REASONING_PROMPTS
        logger.info("Evaluating reasoning on %d prompts", len(test_prompts))

        return self.evaluator.evaluate(
            prompts=test_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            batch_size=batch_size,
        )
