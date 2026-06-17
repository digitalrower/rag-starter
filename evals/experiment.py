import argparse
import logging

from dotenv import load_dotenv

load_dotenv()  # before any langfuse-touching import (E402 exemption needed in pyproject)

from collections.abc import Callable
from typing import Any

from langfuse import Evaluation, get_client

from evals.scorer import score_answer_relevance, score_faithfulness, score_precision
from rag_starter import query
from rag_starter.errors import RAGError

DATASET_NAME_DEFAULT = "rag-starter-eval"

logger = logging.getLogger(__name__)


def make_task(collection: query.Collection) -> Callable[..., dict]:
    """Factory: closes over the Chroma collection so the task matches the SDK signature."""

    def task(*, item: Any, **kwargs: Any) -> dict:
        try:
            generated_result = query.main(collection, item.input)
            chunks_text = [c.text for c in generated_result.chunks]

            response_dict = {
                "answer": generated_result.answer,
                "chunks": chunks_text,
                "sources": generated_result.sources,
            }

            return response_dict

        except RAGError as e:
            logger.error(f"Eval item {item.id} query.main failed: {e}")
            return {
                "error": str(e),
                "answer": "",
                "chunks": [],
                "sources": [],
            }

    return task


def faithfulness_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
) -> Evaluation | list[Evaluation]:

    if output.get("error"):
        return []

    question = input
    generated_answer = output["answer"]
    retrieved_chunks = output["chunks"]
    expected_answer = expected_output

    result = score_faithfulness(question, retrieved_chunks, generated_answer, expected_answer)

    return Evaluation(name="faithfulness", value=result.score, comment=result.reasoning)


def relevance_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
) -> Evaluation | list[Evaluation]:

    if output.get("error"):
        return []

    question = input
    generated_answer = output["answer"]
    expected_answer = expected_output

    result = score_answer_relevance(question, generated_answer, expected_answer)

    return Evaluation(name="relevance", value=result.score, comment=result.reasoning)


def precision_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
) -> Evaluation | list[Evaluation]:

    if output.get("error"):
        return []

    question = input
    retrieved_chunks = output["chunks"]
    expected_answer = expected_output

    result = score_precision(question, retrieved_chunks, expected_answer)

    return Evaluation(name="precision", value=result.score, comment=result.reasoning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a named Langfuse experiment against a seeded dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME_DEFAULT,
        help="Langfuse dataset name to run against (default: rag-starter-eval)",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment run name, shown in the Langfuse Experiments UI (must be unique per run)",
    )

    args = parser.parse_args()

    langfuse = get_client()

    collection = query.get_collection()

    dataset = langfuse.get_dataset(args.dataset)

    result = dataset.run_experiment(
        name=args.name,
        description="Langfuse experiment against a seeded dataset",
        task=make_task(collection),
        evaluators=[faithfulness_evaluator, relevance_evaluator, precision_evaluator],
        # To lower concurrency means fewer simultaneous writes to LF to reduce timeout
        # in order to eliminate possible partial run results. It runs slower, but fine for 40 items.
        max_concurrency=3,
    )

    print(result.format())

    langfuse.flush()


if __name__ == "__main__":
    main()
