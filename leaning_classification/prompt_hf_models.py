from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
import csv
import os
import re
import argparse
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from prompts import ALL_PROMPTS

LABELS_ORDER = ["left", "right", "neutral"]
QUANTIZE_PARAM_THRESHOLD_B = 60


def model_size_in_billions(model_name):
    """Best-effort parse of the parameter count (in billions) from a HF model name, e.g. 'Llama-3.3-70B' -> 70.0."""
    match = re.search(r"(\d+(?:\.\d+)?)b(?:[-_]|$)", model_name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def load_model_and_tokenizer(model_name, cache_dir_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # FIX 1: truncate from the LEFT, not the right.
    # Our prompt template puts the instruction + "ANSWER:" cue *after* {content}.
    # Default truncation_side="right" chops off the end of the sequence when a
    # formatted prompt exceeds max_length, which silently deletes the instruction
    # and generation cue for any sufficiently long article. Truncating from the
    # left instead drops the earliest part of the article and always preserves
    # the task instruction and cue.
    tokenizer.truncation_side = "left"

    size_b = model_size_in_billions(model_name)
    quantize = size_b is not None and size_b >= QUANTIZE_PARAM_THRESHOLD_B

    model_kwargs = dict(
        cache_dir=cache_dir_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantize:
        print(f"Model size ~{size_b}B >= {QUANTIZE_PARAM_THRESHOLD_B}B, loading in 4-bit (bitsandbytes NF4).")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print(model.hf_device_map)

    model.eval()

    first_device_str = next(iter(model.hf_device_map.values()))
    input_device = torch.device(first_device_str)
    return model, tokenizer, input_device


def map_response(text, label_map):
    # response column round-trips through CSV: if every value in a batch is a bare
    # digit (e.g. "2"), pandas infers int64/float64 dtype on re-read, so `text`
    # arrives here as a number (or NaN) rather than a string.
    text = "" if pd.isna(text) else str(text)
    # Strip any visible chain-of-thought (e.g. Qwen3 <think> blocks) before label matching,
    # in case enable_thinking=False wasn't honored by the model's chat template.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_lower = text.lower()
    name_matches = [name for name in LABELS_ORDER if name in text_lower]
    digit_matches = re.findall(r"\b[1-3]\b", text)
    digit_labels = [label_map[int(d)] for d in digit_matches]
    all_labels = name_matches + digit_labels
    if not all_labels:
        return "na"
    return Counter(all_labels).most_common(1)[0][0]


def generate_batch(model, tokenizer, input_device, batch_texts, is_qwen=False, max_new_tokens=10, system_prompt=None):
    """Run one batch of formatted prompts through the model, return decoded responses."""
    chat_template_kwargs = {"enable_thinking": False} if is_qwen else {}
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    batch_prompts = [
        tokenizer.apply_chat_template(
            system_messages + [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for t in batch_texts
    ]
    print(batch_prompts[0])

    inputs = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        # FIX 2: apply_chat_template(tokenize=False) already inserts the model's
        # special tokens (BOS, turn markers, etc). Tokenizing that string again
        # with the default add_special_tokens=True duplicates the BOS token at
        # the start of the sequence. Gemma models are sensitive to malformed
        # special-token sequences at the start of context, and this is a common
        # cause of degenerate repetition-loop generations ("own own own own...").
        add_special_tokens=False,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        remove_invalid_values=True,
        renormalize_logits=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )

    # With padding_side="left", every row is padded to the same total length, so
    # generated tokens for all rows start right after the padded input, not after
    # each row's own (shorter) real-token count.
    input_len = inputs["input_ids"].shape[1]
    responses = []
    for seq in outputs:
        gen_tokens = seq[input_len:]
        responses.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    return responses


def split_reasoning_response(text):
    """Split a raw '...REASONING... ANSWER: X' response into (reasoning, label_text)."""
    text = "" if pd.isna(text) else str(text)
    match = re.search(r"ANSWER:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return text[:match.start()].strip(), match.group(1).strip()
    return text.strip(), text


def compute_and_save_metrics(filename, model_name, prompt_name, label_map, output_dir, run_label=None,
                              system_prompt=None):
    """Map raw responses to labels and save per-label precision/recall/f1, mirroring prompt_gpt.py."""
    results = pd.read_csv(filename)
    results["prediction"] = results["response"].apply(lambda t: map_response(t, label_map))
    results.to_csv(filename, index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        results["gold"], results["prediction"], labels=LABELS_ORDER, zero_division=0,
    )
    rows = []
    for label, p, r, f, s in zip(LABELS_ORDER, precision, recall, f1, support):
        rows.append({"prompt": prompt_name, "label": label, "precision": p, "recall": r, "f1": f, "support": s})

    safe_model_name = model_name.replace("/", "-")
    safe_prompt_name = prompt_name.replace(" ", "_")
    suffix = f"_{run_label}" if run_label else ""
    suffix += "_sysprompt2" if system_prompt else ""
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, f"{safe_model_name}_{safe_prompt_name}{suffix}_summary_metrics.csv"), index=False)
    print(summary_df.to_string(index=False))
    return summary_df


def aggregate_run_metrics(summary_dfs, model_name, prompt_name, output_dir, system_prompt=None):
    """Compute mean/std of precision/recall/f1 across multiple runs (e.g. different data samples)."""
    combined = pd.concat(summary_dfs, ignore_index=True)
    agg = combined.groupby("label")[["precision", "recall", "f1"]].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reindex(LABELS_ORDER).reset_index()

    safe_model_name = model_name.replace("/", "-")
    safe_prompt_name = prompt_name.replace(" ", "_")
    suffix = "_sysprompt2" if system_prompt else ""
    out_path = os.path.join(output_dir, f"{safe_model_name}_{safe_prompt_name}{suffix}_std_across_runs.csv")
    agg.to_csv(out_path, index=False)
    print(f"\n=== Std across {len(summary_dfs)} runs ({prompt_name}) ===")
    print(agg.to_string(index=False))
    return agg


def main(model_name, model, tokenizer, input_device, prompt_name, output_dir="results_hf", batch_size=4,
         reasoning=False, data_path="../../news_article_slant_classification/data/slant_testset.csv", run_label=None,
         system_prompt=None):
    df = pd.read_csv(data_path)

    prompt_template, label_map = ALL_PROMPTS[prompt_name]
    df["formatted_prompt"] = df["content"].apply(lambda c: prompt_template.format(content=c))
    # df["gold"] = df["label"]

    safe_model_name = model_name.replace("/", "-")
    safe_prompt_name = prompt_name.replace(" ", "_")
    suffix = f"_{run_label}" if run_label else ""
    suffix += "_sysprompt2" if system_prompt else ""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{safe_model_name}_{safe_prompt_name}{suffix}.csv")

    # Resume from where we left off if the file already exists
    done_ids = set()
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        done_ids = set(existing["prompt_id"].tolist())
        print(f"Resuming: {len(done_ids)} prompts already processed.")
    else:
        header = ["prompt_id", "response", "reasoning"] if reasoning else ["prompt_id", "response"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    remaining = df[~df["prompt_id"].isin(done_ids)].reset_index(drop=True)
    is_qwen = "qwen" in model_name.lower()
    max_new_tokens = 300 if reasoning else 10

    if remaining.empty:
        print("All prompts already processed.")
    else:
        ids = remaining["prompt_id"].tolist()
        # golds = remaining["gold"].tolist()
        texts = remaining["formatted_prompt"].tolist()

        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc=prompt_name):
                batch_ids = ids[i:i + batch_size]
                # batch_golds = golds[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]

                responses = generate_batch(model, tokenizer, input_device, batch_texts, is_qwen=is_qwen,
                                            max_new_tokens=max_new_tokens, system_prompt=system_prompt)

                with open(filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for pid, response in zip(batch_ids, responses):
                        if reasoning:
                            reasoning_text, label_text = split_reasoning_response(response)
                            writer.writerow([pid, label_text, reasoning_text])
                            continue
                        writer.writerow([pid, response])

        print(f"Done. Responses saved to {filename}")

    # return compute_and_save_metrics(filename, model_name, prompt_name, label_map, output_dir, run_label=run_label,
    #                                  system_prompt=system_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify political slant of news articles with local HuggingFace models.")
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace model name.")
    parser.add_argument("-lp", "--local_path", type=str, default="/data/milanlp/huggingface/hub",
                        help="Local cache directory for HuggingFace models.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("-p", "--prompt", type=str, default=None, choices=list(ALL_PROMPTS.keys()) + ["all"],
                        help="Prompt name from ALL_PROMPTS to apply to each article, or 'all' to run every prompt. "
                             "Ignored if --reasoning is set.")
    parser.add_argument("-o", "--output_dir", type=str, default="results_hf",
                        help="Directory to save responses and metrics.")
    parser.add_argument("--reasoning", action="store_true",
                        help="Use the 'zero_11_reasoning' prompt, which asks the model to output its reasoning "
                             "before the label. Overrides --prompt.")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Optional system message sent to the model before the formatted prompt.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data_dir", type=str, default=None,
                        help="Directory of multiple data CSVs (e.g. leaning_classification/data/) to run the "
                             "same model+prompt over separately, so precision/recall/f1 std across runs can be "
                             "computed. Each CSV must have the same columns as slant_testset.csv. Overrides the "
                             "default single-file testset path.")
    data_group.add_argument("--data_path", type=str, default=None,
                        help="Single data CSV to run (e.g. leaning_classification/data/"
                             "news_articles_leaning_subsampled_neutral.csv). Overrides the default single-file "
                             "testset path.")
    args = parser.parse_args()

    if args.reasoning:
        prompt_names = ["zero_11_reasoning"]
    elif args.prompt:
        prompt_names = list(ALL_PROMPTS.keys()) if args.prompt == "all" else [args.prompt]
    else:
        parser.error("-p/--prompt is required unless --reasoning is set.")

    print(f"Loading model {args.model} from {args.local_path}...")
    model, tokenizer, input_device = load_model_and_tokenizer(args.model, args.local_path)

    if args.data_dir:
        data_paths = sorted(
            os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".csv")
        )
        if not data_paths:
            parser.error(f"No CSV files found in --data_dir {args.data_dir}")
    elif args.data_path:
        data_paths = [args.data_path]
    else:
        data_paths = [None]  # sentinel: fall back to main()'s default single-testset path

    for prompt_name in prompt_names:
        print(f"=== Running prompt: {prompt_name} ===")
        summary_dfs = []
        for data_path in data_paths:
            run_label = os.path.splitext(os.path.basename(data_path))[0] if data_path else None
            kwargs = dict(output_dir=args.output_dir, batch_size=args.batch_size, reasoning=args.reasoning,
                          run_label=run_label, system_prompt=args.system_prompt)
            if data_path:
                kwargs["data_path"] = data_path
                print(f"--- Data sample: {run_label} ---")
            summary_dfs.append(
                main(args.model, model, tokenizer, input_device, prompt_name, **kwargs)
            )

        # if len(summary_dfs) > 1:
        #     aggregate_run_metrics(summary_dfs, args.model, prompt_name, args.output_dir,
        #                            system_prompt=args.system_prompt)