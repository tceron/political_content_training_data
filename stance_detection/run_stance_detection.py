import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from annotate_stance_by_policy import (
    HFModel,
    POLICY_ISSUES,
    STANCE_DETECTION_DIR,
    USER_PROMPT_TEMPLATE,
    call_model,
    parse_full_response,
    prep_policy_examples,
)

OUT_DIR = STANCE_DETECTION_DIR / "out_models_2026"
DEFAULT_MAX_WORKERS = 20


def resolve_id_column(df: pd.DataFrame, csv_path: Path) -> str:
    if "id" in df.columns:
        return "id"
    if "url" in df.columns:
        return "url"
    sys.exit(f"{csv_path}: no 'id' or 'url' column found; columns are {list(df.columns)}")


def build_fieldnames() -> list[str]:
    fieldnames = ["id", "text"]
    for policy_key in POLICY_ISSUES:
        fieldnames += [f"{policy_key}_stance", f"{policy_key}_discussed", f"{policy_key}_raw_response"]
    return fieldnames


def iter_requests(df: pd.DataFrame, examples: pd.DataFrame):
    """Yields (row_idx, policy_key, prompt_text) for every (row, policy) pair."""
    policy_examples = {key: prep_policy_examples(examples, key) for key in POLICY_ISSUES}
    for row_idx, text in enumerate(df.text):
        for policy_key, (policy_label, policy_description) in POLICY_ISSUES.items():
            support_examples, oppose_examples = policy_examples[policy_key]
            prompt_text = USER_PROMPT_TEMPLATE.format(
                text=text,
                policy_label=policy_label,
                policy_description=policy_description,
                support_examples=support_examples,
                oppose_examples=oppose_examples,
            )
            yield row_idx, policy_key, prompt_text




def repair_and_load_ids(out_path: Path) -> set[str]:
    """Drops any trailing malformed row (e.g. left by a killed run) and returns the ids already written."""
    with open(out_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return set()
        rows = []
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                break
            if len(row) == len(header):
                rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    id_idx = header.index("id")
    return {row[id_idx] for row in rows}


def annotate_file(
    client: OpenAI | HFModel,
    model: str,
    csv_path: Path,
    examples: pd.DataFrame,
    limit: int | None,
    out_path: Path,
    max_workers: int,
) -> None:
    """Annotates a single CSV/JSONL file with stance detection for every policy issue, one request per (row, policy) pair."""
    if "jsonl" in str(csv_path):
        df = pd.read_json(csv_path, lines=True, compression="gzip")
    else:
        df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    already_done: set[str] = set()
    if out_path.exists():
        already_done = repair_and_load_ids(out_path)
        df = df[~df["id"].astype(str).isin(already_done)].reset_index(drop=True)
        print(f"{out_path.name}: {len(already_done)} ids already annotated, {len(df)} remaining")

    if df.empty:
        return

    requests = list(iter_requests(df, examples))
    row_ids = dict(enumerate(df["id"]))
    row_texts = dict(enumerate(df.text))
    pending: dict[int, dict[str, str]] = {}

    write_header = not already_done
    with open(out_path, "a" if already_done else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_fieldnames())
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(call_model, client, model, prompt_text): (row_idx, policy_key)
                for row_idx, policy_key, prompt_text in requests
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=csv_path.stem):
                row_idx, policy_key = futures[future]
                pending.setdefault(row_idx, {})[policy_key] = future.result()

                if len(pending[row_idx]) < len(POLICY_ISSUES):
                    continue

                row = {"id": row_ids[row_idx], "text": row_texts[row_idx]}
                for pk, raw in pending.pop(row_idx).items():
                    parsed = parse_full_response(raw)
                    row[f"{pk}_stance"] = parsed["stance"]
                    row[f"{pk}_discussed"] = parsed["discussed"]
                    row[f"{pk}_raw_response"] = raw
                writer.writerow(row)
                f.flush()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run per-policy stance detection over a CSV/JSONL file, "
            "issuing one synchronous request per (row, policy) pair across a thread pool."
        )
    )
    parser.add_argument(
        "-f", "--csv_file", required=True,
        help="CSV/JSONL file to annotate (must have a 'text' column).",
    )
    parser.add_argument(
        "-m", "--model", default="gpt-5.4-mini",
        help="Model name. Names containing '/' (e.g. 'Qwen/Qwen3.6-27B') are loaded as local HuggingFace "
             "models; everything else routes to OpenAI.",
    )
    parser.add_argument(
        "--hf_cache_dir", default="/data/milanlp/huggingface/hub",
        help="Local cache directory for HuggingFace models (only used when --model is a HF repo id).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only annotate the first N rows of each file (for testing).")
    parser.add_argument(
        "-w", "--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Concurrent request workers. Raise this until you start seeing rate-limit retries in the logs.",
    )
    args = parser.parse_args()

    if "/" in args.model:
        print(f"Loading local HuggingFace model {args.model} from {args.hf_cache_dir}...")
        client = HFModel(args.model, cache_dir=args.hf_cache_dir)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY is not set. Add it to leaning_classification/.env or your environment.")
        client = OpenAI(api_key=api_key)

    examples = pd.read_csv(STANCE_DETECTION_DIR / "examples.csv")

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        sys.exit(f"{csv_path} does not exist")

    OUT_DIR.mkdir(exist_ok=True)

    if "gpt" in args.model:
        out_path = OUT_DIR / (csv_path.name.split(".")[0] + ".csv")
    else: 
        out_path = OUT_DIR / (csv_path.name.split(".")[0] + f"_{args.model.replace('/', '_')}.csv")
    print(f"Annotating {csv_path.name} -> {out_path}")
    annotate_file(client, args.model, csv_path, examples, args.limit, out_path, args.max_workers)


if __name__ == "__main__":
    main()
