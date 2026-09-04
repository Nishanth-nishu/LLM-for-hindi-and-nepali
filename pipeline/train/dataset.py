"""JSONL -> flat token-id array -> fixed-length language-modeling blocks."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset


def load_documents(jsonl_path: str | Path, text_field: str = "text") -> list[str]:
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field)
            if text:
                docs.append(text)
    return docs


def _encode_chunk(args: tuple[str, list[str], int]) -> np.ndarray:
    """Runs in a worker process: load the tokenizer once, batch-encode this
    chunk of documents. Module-level (not a closure) so it's picklable for
    ProcessPoolExecutor on every platform, including Windows' spawn start
    method."""
    tokenizer_model, doc_chunk, batch_size = args
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    eos = sp.eos_id()
    arrays: list[np.ndarray] = []
    for start in range(0, len(doc_chunk), batch_size):
        batch = doc_chunk[start : start + batch_size]
        encoded = sp.encode(batch, out_type=int)  # one C++ call for the whole batch
        flat: list[int] = []
        for doc_ids in encoded:
            flat.extend(doc_ids)
            if eos is not None and eos >= 0:
                flat.append(eos)
        arrays.append(np.asarray(flat, dtype=np.int64))
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.int64)


def tokenize_corpus(
    sp_or_path: "spm.SentencePieceProcessor | str | Path",
    documents: list[str],
    batch_size: int = 2000,
    n_workers: int | None = None,
) -> np.ndarray:
    """Concatenate token ids across documents, with the tokenizer's own EOS
    id (if it has one) inserted between documents so a training block never
    silently splices the end of one document onto the start of another
    without a boundary marker.

    Splits `documents` into `n_workers` chunks and tokenizes them in
    parallel worker processes (each with its own SentencePieceProcessor —
    the processor itself isn't picklable, so workers take the model *path*
    and load it locally). At the scale of a ~500M-token corpus this is the
    difference between tokenization taking tens of minutes single-threaded
    versus a few minutes across an 8-core machine. Falls back to
    single-process tokenization for small inputs, where process-pool
    startup overhead would dominate.

    Accepts either a loaded SentencePieceProcessor or a path so callers
    that already have one open (small val/test files) don't pay to reload
    it, while the parallel path can still hand each worker just a path.
    """
    if isinstance(sp_or_path, (str, Path)):
        tokenizer_path = str(sp_or_path)
    else:
        tokenizer_path = sp_or_path.model_file if hasattr(sp_or_path, "model_file") else None

    n_workers = n_workers or min(os.cpu_count() or 1, 8)
    if n_workers <= 1 or len(documents) < 5000 or tokenizer_path is None:
        sp = sp_or_path if not isinstance(sp_or_path, (str, Path)) else spm.SentencePieceProcessor(model_file=str(sp_or_path))
        return _encode_chunk((tokenizer_path or "", documents, batch_size)) if tokenizer_path else _tokenize_serial(sp, documents, batch_size)

    chunk_size = max(1, -(-len(documents) // n_workers))  # ceil div
    chunks = [documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)]
    tasks = [(tokenizer_path, chunk, batch_size) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_encode_chunk, tasks))
    return np.concatenate(results) if results else np.asarray([], dtype=np.int64)


def _tokenize_serial(sp: spm.SentencePieceProcessor, documents: list[str], batch_size: int) -> np.ndarray:
    eos = sp.eos_id()
    arrays: list[np.ndarray] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        encoded = sp.encode(batch, out_type=int)
        flat: list[int] = []
        for doc_ids in encoded:
            flat.extend(doc_ids)
            if eos is not None and eos >= 0:
                flat.append(eos)
        arrays.append(np.asarray(flat, dtype=np.int64))
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.int64)


class TokenBlockDataset(Dataset):
    """Fixed-length next-token-prediction blocks over a flat token array.

    __getitem__(i) -> (x, y):
        x = tokens[i*block_size       : i*block_size + block_size]
        y = tokens[i*block_size + 1   : i*block_size + block_size + 1]

    y is x shifted one position to the right, so cross-entropy(logits, y)
    at position t is exactly "predict token t+1 given tokens <= t".
    """

    def __init__(self, token_ids: np.ndarray, block_size: int):
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"corpus has only {len(token_ids)} tokens, need at least block_size+1={block_size + 1}"
            )
        self.tokens = token_ids
        self.block_size = block_size
        self.n_blocks = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.from_numpy(self.tokens[start:end].copy())
        y = torch.from_numpy(self.tokens[start + 1 : end + 1].copy())
        return x, y


def _cache_path(jsonl_path: Path, tokenizer_model: Path) -> Path:
    return jsonl_path.with_suffix(jsonl_path.suffix + f".{tokenizer_model.stem}.tokens.npy")


def build_dataset(
    jsonl_path: str | Path, tokenizer_model: str | Path, block_size: int, use_cache: bool = True
) -> TokenBlockDataset:
    jsonl_path, tokenizer_model = Path(jsonl_path), Path(tokenizer_model)
    cache = _cache_path(jsonl_path, tokenizer_model)

    if use_cache and cache.exists() and cache.stat().st_mtime >= jsonl_path.stat().st_mtime:
        ids = np.load(cache)
    else:
        docs = load_documents(jsonl_path)
        ids = tokenize_corpus(tokenizer_model, docs)
        if use_cache:
            try:
                np.save(cache, ids)
            except OSError:
                pass  # cache is a pure optimization; a write failure (e.g. read-only fs) shouldn't break training

    return TokenBlockDataset(ids, block_size)
