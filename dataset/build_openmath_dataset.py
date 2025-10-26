from typing import Optional, List
import os
import json
import math
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "unsloth/OpenMathReasoning-mini"
    parquet_path: str = "data/cot-00000-of-00001.parquet"
    output_dir: str = "data/openmath"

    subsample_size: Optional[int] = None
    max_seq_len: Optional[int] = None  # if set, pad/truncate all sequences to this length


def _read_parquet(path: str):
    """Read a parquet file into a pandas.DataFrame using pandas (preferred) or pyarrow as fallback."""
    if pd is not None:
        return pd.read_parquet(path)
    if pq is not None:
        table = pq.read_table(path)
        return table.to_pandas()
    raise RuntimeError("Neither pandas nor pyarrow is available to read parquet files. Please install one of them.")


def _find_columns(df) -> (str, str):
    """Heuristic to pick question and answer columns from the dataframe."""
    cols = [c.lower() for c in df.columns]

    #q_candidates = ["question", "prompt", "input", "problem", "context"]
    #a_candidates = ["answer", "target", "output", "solution", "completion", "final_answer","expected_answer"]

    qcol = "problem"
    acol = "expected_answer"

    # for c in q_candidates:
    #     for orig in df.columns:
    #         if orig.lower() == c:
    #             qcol = orig
    #             break
    #     if qcol:
    #         break

    # for c in a_candidates:
    #     for orig in df.columns:
    #         if orig.lower() == c:
    #             acol = orig
    #             break
    #     if acol:
    #         break

    # # Some datasets use 'input' and 'output' or 'prompt' and 'completion'
    # if qcol is None:
    #     # fallback to first textual column
    #     for orig in df.columns:
    #         if df[orig].dtype == object:
    #             qcol = orig
    #             break

    # if acol is None:
    #     # fallback to a column named like 'answer' else use a blank label
    #     if "answer" in df.columns:
    #         acol = "answer"
    #     else:
    #         # try any other object column different from qcol
    #         for orig in df.columns:
    #             if orig != qcol and df[orig].dtype == object:
    #                 acol = orig
    #                 break

    return qcol, acol

# ...existing code...
def convert_dataset(config: DataProcessConfig):
    # Download parquet
    print(f"Downloading {config.parquet_path} from {config.source_repo}...")
    local_path = hf_hub_download(config.source_repo, config.parquet_path, repo_type="dataset")

    print("Reading parquet file...")
    df = _read_parquet(local_path)

    # Determine splits
    if "split" in df.columns:
        split_values = list(df["split"].unique())
    else:
        # Create 90% train, 10% test split
        total_samples = len(df)
        train_size = int(0.9 * total_samples)
        
        # Shuffle and split the dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["split"] = "train"
        df.loc[train_size:, "split"] = "test"
        split_values = ["train", "test"]
        print(f"Created train/test split: {train_size}/{total_samples-train_size} examples")

    # Find question/answer columns heuristically
    qcol, acol = _find_columns(df)
    if qcol is None:
        raise RuntimeError(f"Could not find a text column for inputs. Available columns: {list(df.columns)}")

    # Build global charset and determine max sequence length across the whole dataframe
    charset = set()
    computed_max_seq = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scanning all examples for vocab/seq_len"):
        q = str(row.get(qcol, "") or "")
        a = str(row.get(acol, "") or "")
        computed_max_seq = max(computed_max_seq, len(q), len(a))
        charset.update(q)
        charset.update(a)

    # Determine final seq_len (use config.max_seq_len if provided)
    if config.max_seq_len is not None:
        if config.max_seq_len < computed_max_seq:
            print(f"Warning: config.max_seq_len ({config.max_seq_len}) < max found ({computed_max_seq}). "
                  f"Sequences will be truncated to {config.max_seq_len}.")
        max_seq_len = config.max_seq_len
    else:
        max_seq_len = computed_max_seq

    # Build char2id mapping (reserve 0 for PAD)
    sorted_chars = sorted(ch for ch in charset if ch != "")  # exclude empty-string if present
    char2id = {ch: idx + 1 for idx, ch in enumerate(sorted_chars)}  # 1..N, 0 is PAD

    # Prepare to save vocab/identifiers once
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    with open(os.path.join(config.output_dir, "char_vocab.json"), "w") as f:
        json.dump({"char2id": char2id, "seq_len": max_seq_len}, f)

    # Encode and save per split using the global max_seq_len and char2id
    for split in split_values:
        if split == "all":
            subdf = df
        else:
            subdf = df[df["split"] == split]

        # Optionally subsample training
        if split == "train" and config.subsample_size is not None:
            total_samples = len(subdf)
            if config.subsample_size < total_samples:
                subdf = subdf.sample(n=config.subsample_size, random_state=42)

        inputs: List[str] = []
        labels: List[str] = []

        for i, row in tqdm(subdf.iterrows(), total=len(subdf), desc=f"Collecting ({split})"):
            q = row.get(qcol, "")
            a = row.get(acol, "")

            inputs.append(str(q))
            labels.append(str(a))

        # Prepare results containers
        results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
        example_id = 0
        puzzle_id = 0
        results["puzzle_indices"].append(0)
        results["group_indices"].append(0)

        for inp, lab in zip(tqdm(inputs, desc=f"Encoding ({split})"), labels):
            # encode to ids and pad/truncate to max_seq_len
            def _encode(s: str):
                arr = np.zeros(max_seq_len, dtype=np.int32)
                for i, ch in enumerate(s[:max_seq_len]):
                    arr[i] = char2id.get(ch, 0)
                return arr

            results["inputs"].append(_encode(inp))
            results["labels"].append(_encode(lab))

            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

            # groups: each puzzle is its own group
            results["group_indices"].append(puzzle_id)

        # Convert to numpy arrays
        results_np = {
            "inputs": np.stack(results["inputs"], axis=0) if len(results["inputs"]) > 0 else np.zeros((0, max_seq_len), dtype=np.int32),
            "labels": np.stack(results["labels"], axis=0) if len(results["labels"]) > 0 else np.zeros((0, max_seq_len), dtype=np.int32),
            "group_indices": np.array(results["group_indices"], dtype=np.int32),
            "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
            "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        }

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=max_seq_len,
            vocab_size=len(char2id) + 1,  # PAD + chars
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=int(len(results_np["group_indices"]) - 1),
            mean_puzzle_examples=1.0,
            total_puzzles=int(len(results_np["group_indices"]) - 1),
            sets=["all"],
        )

        # Save
        save_dir = os.path.join(config.output_dir, str(split))
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        for k, v in results_np.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

        print(f"Saved split '{split}' with {len(inputs)} examples to {save_dir}")
# ...existing code...


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
