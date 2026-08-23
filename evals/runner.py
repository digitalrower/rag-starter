import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before any import that triggers langfuse initialization

from langfuse import get_client, propagate_attributes
from pydantic import TypeAdapter

from evals.scorer import score_answer_relevance, score_faithfulness, score_precision
from rag_starter import query
from rag_starter.client import preflight_env
from rag_starter.errors import RAGError
from rag_starter.models import EvalItem, EvalResult, EvalSummary

# langfuse: initialize langfuse client
langfuse = get_client()

# logging
logger = logging.getLogger(__name__)

# Derived from the model so a category change is a single edit in models.py.
CATEGORIES = [name for name in EvalSummary.model_fields if name != "overall"]

README_TABLE_START = "<!-- eval-table:start -->"
README_TABLE_END = "<!-- eval-table:end -->"


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
            try:
                generated_result = query.main_sync(collection, item.question)
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
            except RAGError as e:
                logger.error(f"Eval item {item.id} failed: {e}")
                item_span.update(level="ERROR", status_message=str(e))
                results.append(
                    EvalResult(
                        id=item.id,
                        category=item.category,
                        question=item.question,
                        expected_answer=item.expected_answer,
                        error=str(e),
                    )
                )

    return results


def write_results(results: list[EvalResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


def print_summary(results: list[EvalResult]) -> None:
    print("\nCategory       Avg Faithfulness    Avg Relevance    Precision@3    Count")
    print("-" * 75)
    all_faith: list[float] = []
    all_relev: list[float] = []
    all_prec: list[float] = []
    errored = [r for r in results if r.error is not None]

    for cat in CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        count = len(cat_results)
        if count == 0:
            continue
        faith_scores = [
            r.faithfulness_score for r in cat_results if r.faithfulness_score is not None
        ]
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
    faith_overall = sum(all_faith) / len(all_faith) if all_faith else None
    relev_overall = sum(all_relev) / len(all_relev) if all_relev else None
    prec_overall = sum(all_prec) / len(all_prec) if all_prec else None
    count_overall = len(all_faith)
    faith_o_display = f"{faith_overall:.2f}" if faith_overall is not None else "N/A"
    relev_o_display = f"{relev_overall:.2f}" if relev_overall is not None else "N/A"
    prec_o_display = f"{prec_overall:.2f}" if prec_overall is not None else "N/A"
    print(
        f"{'OVERALL':<20}{faith_o_display:<20}{relev_o_display:<20}{prec_o_display:<15}{count_overall}"
    )
    print(f"\nErrored items: {len(errored)} / {len(results)}")


def build_summary(results: list[EvalResult]) -> EvalSummary:
    summary: dict[str, object] = {}
    all_faith, all_relev, all_prec = [], [], []
    errored = [r for r in results if r.error is not None]
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        faith_scores = [
            r.faithfulness_score for r in cat_results if r.faithfulness_score is not None
        ]
        relev_scores = [r.relevance_score for r in cat_results if r.relevance_score is not None]
        prec_scores = [r.precision_score for r in cat_results if r.precision_score is not None]
        all_faith.extend(faith_scores)
        all_relev.extend(relev_scores)
        all_prec.extend(prec_scores)
        summary[cat] = {
            "faithfulness": round(sum(faith_scores) / len(faith_scores), 2)
            if faith_scores
            else None,
            "relevance": round(sum(relev_scores) / len(relev_scores), 2) if relev_scores else None,
            "precision": round(sum(prec_scores) / len(prec_scores), 2) if prec_scores else None,
            "count": len(cat_results),
        }
    summary["overall"] = {
        "faithfulness": round(sum(all_faith) / len(all_faith), 2) if all_faith else None,
        "relevance": round(sum(all_relev) / len(all_relev), 2) if all_relev else None,
        "precision": round(sum(all_prec) / len(all_prec), 2) if all_prec else None,
        "count": len(all_faith),
        "errored": len(errored),
    }
    return EvalSummary.model_validate(summary)


def write_summary(summary: EvalSummary, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)


def load_baseline(path: str | Path) -> EvalSummary:
    path = Path(path)
    if not path.exists():
        logger.error(f"No baseline at {path}. Run with --baseline first.")
        sys.exit(1)
    return EvalSummary.model_validate_json(path.read_bytes())


def format_delta(current: float | None, baseline: float | None) -> str:
    if current is None and baseline is None:
        return "-"
    if current is None:
        return "gone"
    if baseline is None:
        return "new"
    diff = current - baseline
    return f"{diff:+.2f}"


def print_comparison(current: EvalSummary, baseline: EvalSummary) -> None:
    print("\nCategory            Faithfulness    Relevance    Precision@3")
    print("-" * 60)
    for name in [*CATEGORIES, "overall"]:
        cur = getattr(current, name)
        base = getattr(baseline, name)
        if cur is None and base is None:
            continue
        cur_faith = cur.faithfulness if cur is not None else None
        cur_relev = cur.relevance if cur is not None else None
        cur_prec = cur.precision if cur is not None else None
        base_faith = base.faithfulness if base is not None else None
        base_relev = base.relevance if base is not None else None
        base_prec = base.precision if base is not None else None
        faith = format_delta(cur_faith, base_faith)
        relev = format_delta(cur_relev, base_relev)
        prec = format_delta(cur_prec, base_prec)
        print(f"{name:<20}{faith:<16}{relev:<13}{prec}")


def format_metric(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def render_table(summary: EvalSummary) -> str:
    lines = [
        "| Category | Avg Faithfulness | Avg Relevance | Precision@3 | Count |",
        "|---|---|---|---|---|",
    ]
    for name in CATEGORIES:
        block = getattr(summary, name)
        if block is None:
            continue
        lines.append(
            f"| {name} | {format_metric(block.faithfulness)} | "
            f"{format_metric(block.relevance)} | "
            f"{format_metric(block.precision)} | {block.count} |"
        )
    o = summary.overall
    lines.append(
        f"| **OVERALL** | **{format_metric(o.faithfulness)}** | "
        f"**{format_metric(o.relevance)}** | "
        f"**{format_metric(o.precision)}** | **{o.count}** |"
    )
    lines.append("")
    lines.append(f"n = {o.count}, {o.errored} errored items.")
    return "\n".join(lines)


def update_readme(summary: EvalSummary, path: str | Path) -> None:
    path = Path(path)
    text = path.read_text()
    start = text.find(README_TABLE_START)
    end = text.find(README_TABLE_END)
    if start == -1 or end == -1:
        logger.error(
            f"Table markers not found in {path}. "
            f"Expected {README_TABLE_START} and {README_TABLE_END}."
        )
        sys.exit(1)
    before = text[: start + len(README_TABLE_START)]
    after = text[end:]
    path.write_text(f"{before}\n{render_table(summary)}\n{after}")


if __name__ == "__main__":
    configure_logging()

    # load_dotenv already ran at module top; this checks the result. A missing
    # Anthropic key aborts here rather than failing all N items identically, and
    # missing Langfuse keys warn now rather than silently writing zero scores.
    preflight_env()

    parser = argparse.ArgumentParser(description="Evaluation Runner for AI datasets")

    # Mutually Exclusive Group (Enforces either/or logic)
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N items",
    )

    # Nargs="+" (Accepts a space-separated list of strings into an array)
    group.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Run only items with these specific IDs (e.g. --ids 027 031)",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Write this run's summary to evals/baseline.json as the tracked baseline",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare this run's summary against evals/baseline.json",
    )

    args = parser.parse_args()

    if (args.baseline or args.compare) and (args.limit is not None or args.ids is not None):
        parser.error("--baseline and --compare cannot be combined with --limit or --ids")

    DATASET_PATH = Path(__file__).parent / "dataset.json"
    RESULTS_PATH = Path(__file__).parent / "results" / "results.json"
    SUMMARY_PATH = Path(__file__).parent / "results" / "summary.json"
    BASELINE_PATH = Path(__file__).parent / "baseline.json"
    README_PATH = Path(__file__).parent.parent / "README.md"

    dataset = load_dataset(DATASET_PATH)

    # Filtering Logic
    if args.ids is not None:
        wanted_ids = set(args.ids)  # convert to set for O(1) lookups
        dataset = [item for item in dataset if item.id in wanted_ids]
        if not dataset:
            logger.error(f"No matching items found for IDs: {args.ids}")
            sys.exit(1)
    elif args.limit is not None:
        dataset = dataset[: args.limit]

    baseline = load_baseline(BASELINE_PATH) if args.compare else None

    collection = query.get_collection()
    graded = run_eval(dataset, collection)
    write_results(graded, RESULTS_PATH)
    summary = build_summary(graded)
    write_summary(summary, SUMMARY_PATH)
    if args.baseline:
        write_summary(summary, BASELINE_PATH)
        update_readme(summary, README_PATH)
        print(f"\nBaseline written to {BASELINE_PATH}")
        print(f"README table updated in {README_PATH}")
    print_summary(graded)

    if args.compare and baseline is not None:
        print_comparison(summary, baseline)

    # langfuse: flush events before the script exits
    langfuse.flush()
