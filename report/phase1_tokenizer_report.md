# Phase 1 — Tokenizer Report

## 1. Objective

Selecting a production SentencePiece vocabulary for the Hindi and Nepali corpora, balancing fertility against vocabulary size and model parameter cost.

## 2. Algorithm

SentencePiece Unigram is used with high Devanagari character coverage and byte fallback.

**Trained tokenizer model (download):** https://drive.google.com/drive/folders/1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY?usp=sharing

## 3. Vocabulary Sweep

The following candidate vocabulary sizes are evaluated:

- 4,000
- 8,000
- 16,000
- 24,000
- 32,000
- 48,000

## 4. Evaluation Metrics

The tokenizer sweep evaluates:

- fertility
- characters/token
- tokens/word
- vocabulary coverage
- tokenization quality
- unknown/byte fallback behavior

## 5. Fertility

Fertility is defined as:

`fertility = number of tokens / number of whitespace-delimited words`

Lower fertility generally indicates more compact tokenization.

However, fertility must be considered together with:

- vocabulary size
- embedding parameter cost
- rare-word handling
- morphological coverage
- multilingual robustness

## 6. Final Vocabulary Selection

The final vocabulary size will be selected based on the tokenizer sweep rather than simply choosing the largest vocabulary.

The decision will balance:

- fertility
- vocabulary size
- model parameter budget
- Hindi coverage
- Nepali coverage
- Devanagari subword quality

## 7. Example

A frequent word may be represented by a single token:

`▁सरकार`

while a less frequent or unseen word may be decomposed into several subword pieces.

Therefore:

**token ≠ word**

## 8. Final Token Count

The final corpus token count will be calculated only after the production tokenizer has been selected.

The calculation is:

`Final Tokens = Σ tokenizer.encode(document).length`

over every document in the final corpus.

## 9. Important Qualification

The Phase 1 acquisition target is approximately:

**400M downloaded + 100M manual = 500M tokens**

but this is not reported as an exact tokenizer measurement until the final tokenizer encoding stage is complete.
