# Scoring-model provenance (FETCHED, not remembered)

The paper's contamination argument rests on these dates, so each one was fetched from the model's
own Hugging Face card on 2026-07-31 and is asserted in `paper/consistency.py`.

| model (Ollama tag) | card | release | stated training cutoff |
|---|---|---|---|
| `qwen2.5:7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | September 2024 (card citation: `month = {September}, year = {2024}`) | NOT STATED on the card |
| `mistral:7b-instruct-v0.2-q4_0` | `mistralai/Mistral-7B-Instruct-v0.2` | 2023; the family's paper `arXiv:2310.06825` is dated October 2023 | NOT STATED on the card |
| `phi3:mini` | `microsoft/Phi-3-mini-4k-instruct` | "Release dates: June, 2024." | "trained on an offline dataset with cutoff date October 2023" |

**The claim the paper is entitled to make.** A model cannot have been trained on data published
after it was itself released, so the release date is a hard upper bound on the training window
regardless of whether a cutoff is published. The most recent of the three was released in September
2024 and the sample window opens 2025-08-01, more than ten months later. The paper says exactly
that and no more; it does not assert a cutoff for the two models that do not publish one.

## The scale-arm model, fetched 2026-08-03

`qwen3.6:latest` resolves to Qwen3.6-35B-A3B (36B total, 3B active, Q4_K_M). Release date **2026-04-16**,
verbatim from the vendor's own repository README (`github.com/QwenLM/Qwen3.6`):
"2026-04-16: Qwen3.6-35B-A3B is now availabe on Hugging Face Hub and ModelScope." (sic).

**This is INSIDE the sample window** (2025-08-01 to 2026-07-30), about eight and a half months in.
The release-date bound therefore does NOT cover this model. Filings from the portion of the sample
that PRECEDES 2026-04-16, roughly the first two thirds, may have been available to its training
process; filings after it cannot have been. The paper says exactly this and claims no direction for
the possible bias, because none is established: exposure could sharpen or degrade the association,
and a directional claim would be unsupported.
