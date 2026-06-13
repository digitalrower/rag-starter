import sys
import time

from dotenv import load_dotenv

# load_dotenv must run before the langfuse import below: get_client() reads
# LANGFUSE_* env vars at import/init time. Same pattern as runner.py. This is
# why this file has an E402 exemption in pyproject.toml (imports not at top).
load_dotenv()

from pathlib import Path

from langfuse import get_client
from pydantic import TypeAdapter

from rag_starter.models import EvalItem

# Single source for the name: used in create_dataset, create_dataset_item,
# and the verify fetch.
DATASET_NAME = "rag-starter-eval"

DATASET_PATH = Path(__file__).parent.parent / "evals" / "dataset.json"


# Inlined rather than imported from evals.runner: importing runner drags in
# chromadb/scorer/query and fires runner's module-level load_dotenv, all for
# three lines. Deviation from diagram d3 noted in the week file.
def load_dataset(path: Path) -> list[EvalItem]:
    adapter = TypeAdapter(list[EvalItem])
    return adapter.validate_json(path.read_bytes())


def seed() -> None:
    items = load_dataset(DATASET_PATH)
    langfuse = get_client()

    langfuse.create_dataset(name=DATASET_NAME)

    for item in items:
        # id=item.id is the upsert key: re-running this script overwrites
        # existing items instead of duplicating them (verified empirically,
        # second run kept the count at the same number).
        # Field mapping (locked in the W7E Friday design):
        #   question -> input (raw string; experiment.py task reads item.input)
        #   expected_answer -> expected_output (evaluators receive it)
        #   everything else -> metadata (None values pass through as-is)
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item.id,
            input=item.question,
            expected_output=item.expected_answer,
            metadata={"category": item.category, "pair_id": item.pair_id, "notes": item.notes},
        )

    # Langfuse SDK batches writes in the background; flush before fetching
    # back or the verify count can read stale.
    langfuse.flush()

    expected = len(items)
    actual = -1
    for attempt in range(2):
        fetched = langfuse.get_dataset(DATASET_NAME)
        actual = len(fetched.items)
        if actual == expected:
            print(f"Seed OK: {actual}/{expected} items in '{DATASET_NAME}'")
            return
        if attempt == 0:
            # Exactly one retry: server-side ingestion can lag the fetch
            # briefly even after flush. Still wrong after 3s = real failure.
            time.sleep(3)

    # Nonzero exit so a wrapping shell or CI step treats mismatch as failure.
    print(f"Seed FAILED: expected {expected}, Langfuse returned {actual}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    seed()
