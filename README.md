# Zero-Resource Detection of Persuasion Techniques in European Portuguese

Span-level detection of **23 persuasion techniques** in European Portuguese news
articles under a strict **zero-resource** setting (no annotated training data in
the target language), built on the **CLEF-2024 CheckThat! Lab Task 3** data.

The system learns from six source languages and transfers to Portuguese through
cross-lingual representations and machine-translated data, selecting the transfer
source language by an embedding-based proximity analysis. On the Portuguese test
set it outperforms the CLEF-2024 competition winner and the organisers'
post-competition system (F1-micro).

- 🤖 **Model:** released on the Hugging Face Hub (see Section 11)
- 🌍 **Task / data:** CLEF-2024 CheckThat! Lab Task 3 (see Section 8)

---

## 1. Project Overview

The detector is a fine-tuned **XLM-RoBERTa** encoder with a **multi-label sigmoid
head** that emits, per token, independent begin/inside (BIO) outputs for every
technique, so co-occurring and overlapping techniques are predicted directly.
Training follows a two-stage pipeline:

- **Stage 1** - multilingual training on all six source languages (English,
  Italian, French, German, Polish, Russian) plus machine-translated Portuguese.
- **Stage 2** - fine-tuning on Italian-to-Portuguese translated data alone, the
  source language identified as closest to Portuguese by the embedding analysis.

An interactive **Streamlit** demo (`app/app_streamlit.py`) runs the trained model
on arbitrary Portuguese text.

## 2. Key Features

- Per-token **multi-label** formulation (46 outputs = 23 techniques × B/I) that
  handles overlapping and nested spans without a separate span decoder.
- **Embedding-guided** selection of the transfer source language.
- **Zero-resource**: no Portuguese training or development labels are used at any
  point; the Portuguese data is used only for evaluation.
- Full **translate-train** pipeline (MT + span projection) for IT/FR/RU → PT.
- Released model weights, reproducible training/evaluation scripts, and a demo app.

## 3. Project Status

The system is stable and suitable for **research use**. The trained model,
evaluation scripts, and the embedding-proximity analysis are provided to
reproduce the results reported in the paper.

## 4. Technology Stack

- **Language:** Python 3.10
- **Core ML:** PyTorch, Hugging Face Transformers (XLM-RoBERTa)
- **Embeddings:** sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Translation / alignment:** Google Translate, DeepL, LinguAligner (span projection)
- **App:** Streamlit
- **Utilities:** pandas, openpyxl (results), Weights & Biases (training logs)
- **Hardware:** experiments run on a single GPU with fp16 mixed precision

## 5. Dependencies

Install the core environment from the repository root:

```bash
pip install -r requirements.txt
```

The demo app has its own lighter dependency file:

```bash
pip install -r app/requirements_streamlit.txt
```

## 6. Repository Structure

```
.
├── app/
│   ├── app_streamlit.py                  # Streamlit application (Frontend & Backend)
│   └── requirements_streamlit.txt        # Streamlit environment dependencies
├── best_model/                           # Best trained model (IT→PT only, Stage 2)
│           ├── model.pt                  # NOT included — see Section 11 (Hugging Face)
│           ├── config.json
│           ├── multilabel_config.json
│           └── tokenizer files
├── embeddings/
│   ├── embedding_analysis.py             # Document embedding analysis
│   └── embeddings_analysis_text_only.py  # Document embedding analysis (text only)
├── models_improved/
│   ├── train_model_base_v4.py            # Stage 1: multilingual base training
│   ├── train_it_pt_finetune_v4_it.py     # Stage 2: Portuguese fine-tuning
│   └── testing.py                        # Inference + evaluation (Algorithm 1)
├── results/
│   └── results_persuasion_detection.xlsx # All results across models
├── translation/
│   ├── translate_it_to_pt.py             # IT→PT translation (Google + LinguAligner)
│   ├── translate_fr_to_pt.py             # FR→PT translation (Google + LinguAligner)
│   └── translate_ru_to_pt.py             # RU→PT translation (DeepL + LinguAligner)
├── .gitignore
├── README.md
└── requirements.txt
```

## 7. Usage

The pipeline runs in four steps. Set the CLEF-2024 data path inside the scripts
(or via the config at the top of each file) before running.

**1. Build translate-train data (MT + span projection):**

```bash
python translation/translate_it_to_pt.py   # IT → PT  (used for the final model)
python translation/translate_fr_to_pt.py   # FR → PT  (ablation)
python translation/translate_ru_to_pt.py   # RU → PT  (ablation)
```

**2. Language-proximity analysis (nearest cross-lingual neighbours):**

```bash
python embeddings/embedding_analysis.py            # text + technique-distribution embeddings
python embeddings/embeddings_analysis_text_only.py # text-only variant
```

**3. Train — Stage 1 (multilingual) then Stage 2 (IT→PT fine-tuning):**

```bash
python models_improved/train_model_base_v4.py       # Stage 1
python models_improved/train_it_pt_finetune_v4_it.py # Stage 2 (final model)
```

**4. Inference and evaluation on the Portuguese test set:**

```bash
python models_improved/testing.py   # Algorithm 1: inference + partial-span-matching F1
```

**Run the demo app:**

```bash
streamlit run app/app_streamlit.py
```

> ℹ️ **Key hyperparameters:** XLM-RoBERTa encoder; dropout 0.15; sliding windows of
> 512 tokens, stride 256; per-output `pos_weight` (neg/pos ratio, capped to
> [10, 100]); Stage 1 lr 3e-5, effective batch 16, up to 10 epochs (early stopping,
> patience 3), 6% warmup, weight decay 0.01, fp16; Stage 2 lr 1e-5, up to 15 epochs;
> decision threshold τ = 0.70 (selected on the Italian development set).

## 8. Dataset Description

> ⚠️ **Access:** This repository does **not** redistribute the CLEF-2024 CheckThat!
> Lab Task 3 data. Obtain it from the official CLEF-2024 CheckThat! Lab Task 3
> channels and respect its terms of use.

| Split        | Languages                                     |
|--------------|-----------------------------------------------|
| Training     | EN, IT, FR, DE, PL, RU                         |
| Development  | EN, IT, FR, DE, PL, RU                         |
| Test         | European Portuguese — 104 documents, 1,727 spans |

Portuguese has **no** training or development data: it provides only a test set,
which defines the zero-resource setting. The `translation/` scripts produce the
machine-translated Portuguese used in training (IT→PT: 302 docs / 4,735 spans;
FR→PT: 208 / 4,166; RU→PT: 190 / 3,096).

**Reference:** Hasanain, M., et al. *Overview of the CLEF-2024 CheckThat! Lab
Task 3 on Persuasion Techniques Detection.* CLEF Working Notes, 2024.

## 9. Architecture

<div align="center">
  <em>XLM-RoBERTa encoder → multi-label sigmoid head (46 outputs) → BIO decoding → spans</em>
</div>

- **Encoder:** XLM-RoBERTa, shared across all techniques.
- **Head:** dropout (p = 0.15) + linear projection 768 → 46, sigmoid activation; a
  token is assigned a (technique, position) label when its probability exceeds the
  global threshold τ.
- **Long documents:** 94% of articles exceed the 512-token window, so each document
  is processed in overlapping windows (stride 256) and probabilities are averaged.
- **Transfer-language selection:** nearest cross-lingual neighbours of the 104
  Portuguese documents (`embeddings/`) identify Italian as the closest source.
- **Inference (Algorithm 1):** sliding-window inference, probability averaging,
  thresholding, and span grouping in `models_improved/testing.py`.

## 10. Results

Span-level F1 on the CLEF-2024 CheckThat! Task 3 **Portuguese test set**
(full breakdown in `results/results_persuasion_detection.xlsx`):

| System                                   | F1-micro | F1-macro |
|------------------------------------------|:--------:|:--------:|
| **Ours** (multi-label sigmoid, IT→PT)    | **0.134**| 0.103    |
| PersuasionMultiSpan (organisers)         | 0.132    | 0.120    |
| UniBO (CLEF-2024 winner)                  | 0.107    | 0.073    |
| mBERT baseline (BIO softmax)             | 0.057    | 0.017    |
| XLM-R baseline (BIO softmax)             | 0.043    | 0.005    |
| Gemini 2.5 Flash (few-shot)              | 0.037    | 0.040    |

Absolute scores are low because the task is hard: human inter-annotator agreement
(Krippendorff's α) is 0.404 before curation, and 0.587 for Portuguese. Scores are
best read relative to the competing systems and this task-specific ceiling.

## 11. Resources

- 🤖 **Trained model (weights):** released on the Hugging Face Hub. The `model.pt`
  under `outputs_v4/.../best_model/` is **not** committed to this repository.
- 🌍 **Sentence embeddings:** [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- 🔗 **Span projection:** LinguAligner

## 12. License

## 13. Acknowledgments

Built on the CLEF-2024 CheckThat! Lab Task 3 dataset and resources. Developed at
INESC TEC / University of Porto.

---

*Last updated: October 2026.*
