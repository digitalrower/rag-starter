import logging
import sys

from anthropic import APIError
from anthropic.types import TextBlock
from langfuse import get_client
from pydantic import ValidationError

from rag_starter.client import get_anthropic_client
from rag_starter.errors import ResponseParseError, ScoringError
from rag_starter.models import ScoreResult

# langfuse: initialize langfuse client
langfuse = get_client()

logger = logging.getLogger(__name__)


def score_precision(
    question: str, retrieved_chunks: list[str], expected_answer: str
) -> ScoreResult:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="scorer_precision",
        input={
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "expected_answer": expected_answer,
        },
    ) as span:
        context = "\n\n".join(retrieved_chunks)
        client = get_anthropic_client()

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0,
                system=(
                    "You are an evaluator. Given the following context, question, "
                    "and expected answer, "
                    "score whether at least one of the retrieved chunks contains sufficient "
                    "information to answer the question. "
                    "Score 1 if at least one chunk contains information that could answer "
                    "the question. "
                    "Score 0 if none of the chunks contain relevant information. "
                    "Sufficient information means the chunks don't need the exact answer "
                    "word-for-word, just enough that a reasonable answer could be "
                    "generated from them. "
                    'Return JSON only: {"score": int, "reasoning": str}'
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\nQuestion: {question}\n\n"
                            "Answer: {expected_answer}"
                        ),
                    }
                ],
            )
        except APIError as e:
            error_msg = f"Couldn't reach Claude: {e}"
            logger.error(f"Anthropic API call failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ScoringError(error_msg) from e

        block = response.content[0]
        raw = block.text if isinstance(block, TextBlock) else ""

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed_result = ScoreResult.model_validate_json(raw)
        except ValidationError as e:
            error_msg = f"Couldn't parse response: {e}"
            logger.error(f"Parse response failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ResponseParseError(error_msg, raw=raw) from e

        span.update(output=parsed_result.model_dump())
        logger.info(f"Scoring complete: scorer_precision score={parsed_result.score}")
        return parsed_result


def score_faithfulness(
    question: str,
    retrieved_chunks: list[str],
    generated_answer: str,
    expected_answer: str,
) -> ScoreResult:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="scorer_faithfulness",
        input={
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
        },
    ) as span:
        context = "\n\n".join(retrieved_chunks)
        client = get_anthropic_client()
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0,
                system=(
                    "You are an evaluator. Given the following context, question, and answer, "
                    "score the answer for faithfulness on a scale of 1-5. "
                    "Faithfulness = the answer only uses information present "
                    "in the provided context. "
                    "Score 5 if fully grounded, 1 if it contains hallucinated "
                    "facts not in context. "
                    'Return JSON only: {"score": int, "reasoning": str}'
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\nQuestion: {question}\n\n"
                            "Answer: {generated_answer}"
                        ),
                    }
                ],
            )
        except APIError as e:
            error_msg = f"Couldn't reach Claude: {e}"
            logger.error(f"Anthropic API call failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ScoringError(error_msg) from e

        block = response.content[0]
        raw = block.text if isinstance(block, TextBlock) else ""

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed_result = ScoreResult.model_validate_json(raw)
        except ValidationError as e:
            error_msg = f"Couldn't parse response: {e}"
            logger.error(f"Parse response failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ResponseParseError(error_msg, raw=raw) from e

        span.update(output=parsed_result.model_dump())
        logger.info(f"Scoring complete: scorer_faithfulness score={parsed_result.score}")
        return parsed_result


def score_answer_relevance(
    question: str, generated_answer: str, expected_answer: str
) -> ScoreResult:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="scorer_relevance",
        input={
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
        },
    ) as span:
        client = get_anthropic_client()

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                temperature=0,
                system=(
                    "You are an evaluator. Given the following question, generated answer "
                    "and expected answer, score the answer for answer relevance on a scale of 1-5. "
                    "Answer Relevance = a measure of whether the generated answer actually "
                    "addresses the question that was asked. "
                    "Score 5 if the answer fully and directly addresses the question, "
                    "1 if the answer is off-topic, evasive, or fails to address what was asked. "
                    'Return JSON only: {"score": int, "reasoning": str}'
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Generated Answer: {generated_answer}\n\n"
                            f"Expected Answer: {expected_answer}"
                        ),
                    }
                ],
            )
        except APIError as e:
            error_msg = f"Couldn't reach Claude: {e}"
            logger.error(f"Anthropic API call failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ScoringError(error_msg) from e

        block = response.content[0]
        raw = block.text if isinstance(block, TextBlock) else ""

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed_result = ScoreResult.model_validate_json(raw)
        except ValidationError as e:
            error_msg = f"Couldn't parse response: {e}"
            logger.error(f"Parse response failed: {error_msg}")
            span.update(level="ERROR", status_message=error_msg)
            raise ResponseParseError(error_msg, raw=raw) from e

        span.update(output=parsed_result.model_dump())
        logger.info(f"Scoring complete: scorer_relevance score={parsed_result.score}")
        return parsed_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    chunks = ["Agents use skills to perform tasks. A skill is a reusable capability."]

    cases = [
        {
            "label": "faithful",
            "question": "What are agent skills?",
            "answer": "Agent skills are reusable capabilities that agents use to perform tasks.",
            "expected": "Agent skills are reusable capabilities.",
        },
        {
            "label": "unfaithful",
            "question": "What are agent skills?",
            "answer": "Agent skills include neural interfaces and quantum reasoning modules.",
            "expected": "Agent skills are reusable capabilities.",
        },
        {
            "label": "i_dont_know",
            "question": "What are agent skills?",
            "answer": "I don't have enough information to answer that question.",
            "expected": "Agent skills are reusable capabilities.",
        },
    ]

    with langfuse.start_as_current_observation(as_type="span", name="offline_eval_run"):
        for case in cases:
            result = score_faithfulness(
                case["question"],
                chunks,
                case["answer"],
                case["expected"],
            )
            print(f"[{case['label']}] score={result.score} | {result.reasoning}")

    langfuse.flush()
