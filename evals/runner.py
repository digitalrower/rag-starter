import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() # must run before any import that triggers langfuse initialization

from langfuse import get_client, propagate_attributes
from pydantic import TypeAdapter

from evals.scorer import score_answer_relevance, score_faithfulness, score_precision
from rag_starter import query
from rag_starter.models import EvalItem, EvalResult

# langfuse: initialize langfuse client
langfuse = get_client()

# logging
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def load_dataset(path: Path) -> list[EvalItem]:
    adapter = TypeAdapter(list[EvalItem])
    return adapter.validate_json(path.read_bytes())

def run_eval(dataset: list[EvalItem], collection: query.Collection) -> list[EvalResult]:
    run_id = f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    results = []
    #for item in dataset[26:27]:    # use for smoke test to save on tokens; $ python -m evals.runner
    for item in dataset:
        with (
            propagate_attributes(session_id=run_id, tags=["eval", item.category]),
            langfuse.start_as_current_observation(
                as_type="span",
                name="eval_item",
                input={
                    "id": item.id,
                    "category": item.category,
                    "question": item.question,
                },
            ) as item_span,
        ):
            generated_result = query.main(collection, item.question)
            chunks_text = [c.text for c in generated_result.chunks]

            faithfulness = score_faithfulness(
                item.question,
                chunks_text,
                generated_result.answer,
                item.expected_answer,
            )
            relevance = score_answer_relevance(
                item.question, generated_result.answer, item.expected_answer
            )
            precision = score_precision(item.question, chunks_text, item.expected_answer)

            # langfuse: emit one score object per metric, attached to this item's trace
            trace_id = generated_result.trace_id
            langfuse.create_score(
                trace_id=trace_id,
                name="faithfulness",
                value=int(faithfulness.score),
                data_type="NUMERIC",
                comment=str(faithfulness.reasoning),
            )
            langfuse.create_score(
                trace_id=trace_id,
                name="relevance",
                value=int(relevance.score),
                data_type="NUMERIC",
                comment=str(relevance.reasoning),
            )
            langfuse.create_score(
                trace_id=trace_id,
                name="precision",
                value=int(precision.score),
                data_type="NUMERIC",
                comment=str(precision.reasoning),
            )

            item_span.update(
                output={
                    "faithfulness": int(faithfulness.score),
                    "relevance": int(relevance.score),
                    "precision": int(precision.score),
                }
            )

            results.append(
                EvalResult(
                    id=item.id,
                    category=item.category,
                    question=item.question,
                    expected_answer=item.expected_answer,
                    actual_answer=generated_result.answer,
                    faithfulness_score=faithfulness.score,
                    faithfulness_reasoning=faithfulness.reasoning,
                    relevance_score=relevance.score,
                    relevance_reasoning=relevance.reasoning,
                    precision_score=precision.score,
                    precision_reasoning=precision.reasoning,
                    sources=generated_result.sources,
                )
            )

    return results


def write_results(results: list[EvalResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


def print_summary(results: list[EvalResult]) -> None:
    categories = ["happy_path", "edge_case", "adversarial", "bias_paired"]
    print("\nCategory       Avg Faithfulness    Avg Relevance    Precision@3    Count")
    print("-" * 75)
    all_faith: list[float] = []
    all_relev: list[float] = []
    all_prec: list[float] = []

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        count = len(cat_results)
        if count == 0:
            continue
        faith_scores = [r.faithfulness_score for r in cat_results if r.faithfulness_score is not None]
        relev_scores = [r.relevance_score for r in cat_results if r.relevance_score is not None]
        prec_scores = [r.precision_score for r in cat_results if r.precision_score is not None]
        faith_avg = sum(faith_scores) / len(faith_scores) if faith_scores else None
        relev_avg = sum(relev_scores) / len(relev_scores) if relev_scores else None
        prec_avg = sum(prec_scores) / len(prec_scores) if prec_scores else None
        all_faith.extend(faith_scores)
        all_relev.extend(relev_scores)
        all_prec.extend(prec_scores)
        faith_display = f"{faith_avg:.2f}" if faith_avg is not None else "N/A"
        relev_display = f"{relev_avg:.2f}" if relev_avg is not None else "N/A"
        prec_display = f"{prec_avg:.2f}" if prec_avg is not None else "N/A"
        print(f"{cat:<20}{faith_display:<20}{relev_display:<20}{prec_display:<15}{count}")


    print("-" * 75)
    faith_overall = sum(all_faith) / len(all_faith)
    relev_overall = sum(all_relev) / len(all_relev)
    prec_overall = sum(all_prec) / len(all_prec)
    count_overall = len(all_faith)
    print(
        f"{'OVERALL':<20}{faith_overall:<20.2f}"
        f"{relev_overall:<20.2f}{prec_overall:<15.2f}{count_overall}"
    )


def write_summary(results: list[EvalResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = ["happy_path", "edge_case", "adversarial", "bias_paired"]
    summary: dict[str, object] = {}
    all_faith, all_relev, all_prec = [], [], []
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        faith_scores = [r.faithfulness_score for r in cat_results if r.faithfulness_score is not None]
        relev_scores = [r.relevance_score for r in cat_results if r.relevance_score is not None]
        prec_scores = [r.precision_score for r in cat_results if r.precision_score is not None]
        all_faith.extend(faith_scores)
        all_relev.extend(relev_scores)
        all_prec.extend(prec_scores)
        summary[cat] = {
            "faithfulness": round(sum(faith_scores) / len(faith_scores), 2),
            "relevance": round(sum(relev_scores) / len(relev_scores), 2),
            "precision": round(sum(prec_scores) / len(prec_scores), 2),
            "count": len(cat_results),
        }
    summary["overall"] = {
        "faithfulness": round(sum(all_faith) / len(all_faith), 2),
        "relevance": round(sum(all_relev) / len(all_relev), 2),
        "precision": round(sum(all_prec) / len(all_prec), 2),
        "count": len(all_faith),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    configure_logging()

    DATASET_PATH = Path(__file__).parent / "dataset.json"
    dataset = load_dataset(DATASET_PATH)
    collection = query.get_collection()
    graded = run_eval(dataset, collection)
    RESULTS_PATH = Path(__file__).parent / "results" / "results.json"
    SUMMARY_PATH = Path(__file__).parent / "results" / "summary.json"
    write_results(graded, RESULTS_PATH)
    write_summary(graded, SUMMARY_PATH)
    print_summary(graded)

    # langfuse: flush events before the script exits
    langfuse.flush()
