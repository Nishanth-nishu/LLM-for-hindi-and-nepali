"""Tokenized (prompt, answer) pairs for reasoning finetuning.

Answer-only loss masking: the model must never be penalized for failing
to predict the (fixed, given) question text, only for predicting the
answer that follows the language's answer delimiter (Hindi "उत्तर:",
Nepali "जवाफ:"). This reuses GPTLanguageModel.forward's existing
`ignore_index=-100` support in its cross-entropy loss (see
pipeline/model/transformer.py) -- no model change needed, only how the
training targets are built here.

Sequences are right-padded to a fixed length with the tokenizer's pad_id.
Padding is safe for a causal decoder even without an explicit attention
mask: padding tokens are placed strictly after all real content, so the
causal mask already prevents them from influencing any real position's
logits, and their loss positions are masked out the same way prompt
positions are.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class ReasoningExampleDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer_model: str | Path,
        delimiter: str,
        max_len: int = 128,
    ):
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        self.delimiter = delimiter
        self.max_len = max_len
        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()

        self.examples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def encode(self, ex: dict) -> tuple[list[int], list[int]]:
        """Returns (x, y): x is the (padded, truncated) input sequence, y is
        the next-token target with -100 on every position whose target
        token belongs to the prompt (or padding), so only answer-token
        predictions contribute to the loss."""
        prompt_ids = self.sp.encode(f'{ex["prompt"]} {self.delimiter} ', out_type=int)
        answer_ids = self.sp.encode(ex["answer"], out_type=int)
        if self.eos_id is not None and self.eos_id >= 0:
            answer_ids = answer_ids + [self.eos_id]

        full = prompt_ids + answer_ids
        full = full[: self.max_len + 1]  # +1 because x,y are built from consecutive positions
        n_prompt = min(len(prompt_ids), len(full))

        x = full[:-1] if len(full) > 1 else full
        y = [
            full[i + 1] if (i + 1) >= n_prompt else IGNORE_INDEX
            for i in range(len(full) - 1)
        ]

        pad_needed = self.max_len - len(x)
        if pad_needed > 0:
            x = x + [self.pad_id] * pad_needed
            y = y + [IGNORE_INDEX] * pad_needed
        return x, y

    def __getitem__(self, idx: int):
        x, y = self.encode(self.examples[idx])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
