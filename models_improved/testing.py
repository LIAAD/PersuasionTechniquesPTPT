"""
Testing script for CLEF-2024 CheckThat! Lab Task 3 — Persuasion Technique Detection
Compatible with the multi-label sigmoid model (v3b).

Produces predictions as JSON: list of {id, text, spans: [{start, end, label}]}
matching the gold format for evaluation with the official scorer.

Key design decisions based on the paper's evaluation metric:
  - Partial span overlap gives partial credit (≥50% overlap of gold → full credit)
  - Predictions >4x the gold span length get zero credit → avoid very long spans
  - Greedy matching by max S(p,g) → better to have more precise shorter spans
  - Multi-label: same text region can have multiple technique labels

Threshold selection:
  The global decision threshold is selected on a SOURCE-LANGUAGE development set
  (Italian, the closest language to Portuguese in the embedding space) and then
  applied once to the Portuguese test set. It is never selected on the PT gold,
  which would use the test set for a modelling decision.
"""

import os, glob, re, json, sys
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIG 
# ============================================================

MODEL_DIR = "outputs_v4/pt_finetuned_only_it/best_model"    
TEST_DIR  = "test-articles-subtask-3-pt"             
GOLD_JSON = "json/checkthat24_pt.json"              
CALIB_JSON = "json/dev_it.json"   # source-language dev for threshold selection (must NOT be the PT gold)

OUT_PREDICTIONS_JSON = "outputs_v4/predictions_v4it.json"   
OUT_PREDICTIONS_TSV  = "outputs_v4/predictions_v4it.tsv"   

# Inference config
MAX_LENGTH    = 512
CHUNK_OVERLAP = 0.5   # 50% overlap as in the organizers' system
THRESHOLD     = 0.3   # fallback only; the operating threshold is selected on CALIB_JSON

# Thresholds swept when selecting the operating threshold
THRESHOLD_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

# ============================================================
# LABELS
# ============================================================

LABELS = [
    "Loaded_Language", "Name_Calling-Labeling", "Repetition", "Slogans",
    "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
    "Appeal_to_Authority", "Appeal_to_Values", "Doubt", "Exaggeration-Minimisation",
    "Guilt_by_Association", "False_Dilemma-No_Choice", "Straw_Man", "Red_Herring",
    "Whataboutism", "Obfuscation-Vagueness-Confusion", "Appeal_to_Time",
    "Conversation_Killer", "Appeal_to_Popularity", "Appeal_to_Hypocrisy",
    "Consequential_Oversimplification", "Questioning_the_Reputation",
]
NUM_LABELS = len(LABELS)
NUM_OUTPUTS = NUM_LABELS * 2
label2idx = {l: i for i, l in enumerate(LABELS)}

# ============================================================
# DEVICE
# ============================================================

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"Device: {device_str}")

# ============================================================
# MODEL DEFINITION (must match training)
# ============================================================

class XLMRMultiLabelNER(nn.Module):
    def __init__(self, model_name, num_outputs, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, num_outputs)
        self.num_outputs = num_outputs
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight] * num_outputs))
        else:
            self.pos_weight = None

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return {"logits": logits}

    @classmethod
    def load_pretrained(cls, path):
        with open(os.path.join(path, "multilabel_config.json")) as f:
            config = json.load(f)
        # No pos_weight needed at inference
        model = cls(config["model_name"], config["num_outputs"], pos_weight=None)
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        # Filter out pos_weight from state dict if present
        state = {k: v for k, v in state.items() if "pos_weight" not in k}
        model.load_state_dict(state, strict=False)
        return model

# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model from {MODEL_DIR}...")
model = XLMRMultiLabelNER.load_pretrained(MODEL_DIR)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
print("Model loaded.")

# ============================================================
# SLIDING WINDOW INFERENCE (token-level, 50% overlap)
# ============================================================

def predict_document(text, threshold=THRESHOLD):
    """
    Run inference on a full document using sliding windows.
    Returns list of (char_start, char_end, label, score) spans.
    """
    # Tokenize full document without special tokens
    full_enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    full_ids = full_enc["input_ids"]
    full_offsets = full_enc["offset_mapping"]
    total_tokens = len(full_ids)

    usable_len = MAX_LENGTH - 2  # space for BOS + EOS
    stride = max(1, int(usable_len * (1 - CHUNK_OVERLAP)))

    # Create windows
    if total_tokens <= usable_len:
        windows = [(0, total_tokens)]
    else:
        windows = []
        start = 0
        while start < total_tokens:
            end = min(start + usable_len, total_tokens)
            windows.append((start, end))
            if end == total_tokens:
                break
            start += stride

    # Per-token probability accumulation (for overlap averaging)
    # Shape: (total_tokens, NUM_OUTPUTS)
    prob_sum = np.zeros((total_tokens, NUM_OUTPUTS), dtype=np.float64)
    prob_count = np.zeros(total_tokens, dtype=np.float64)

    bos = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos = tokenizer.eos_token_id or tokenizer.sep_token_id

    for (win_start, win_end) in windows:
        chunk_ids = full_ids[win_start:win_end]

        input_ids = torch.tensor([[bos] + chunk_ids + [eos]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"][0].cpu().numpy()  # (seq_len, NUM_OUTPUTS)

        # Apply sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Skip BOS (index 0) and EOS (last index)
        chunk_probs = probs[1:-1]  # (chunk_len, NUM_OUTPUTS)

        for i, token_idx in enumerate(range(win_start, win_end)):
            prob_sum[token_idx] += chunk_probs[i]
            prob_count[token_idx] += 1

    # Average probabilities across overlapping windows
    prob_count[prob_count == 0] = 1
    avg_probs = prob_sum / prob_count[:, None]

    # Extract active predictions per token
    token_preds = []  # list of (token_idx, label_name, tag_type, score)
    for t_idx in range(total_tokens):
        for li, label in enumerate(LABELS):
            b_prob = avg_probs[t_idx, 2 * li]
            i_prob = avg_probs[t_idx, 2 * li + 1]

            if b_prob > threshold or i_prob > threshold:
                tag = "B" if b_prob >= i_prob else "I"
                score = float(max(b_prob, i_prob))
                token_preds.append((t_idx, label, tag, score))

    # Group consecutive B/I tokens into spans, per label
    spans = group_tokens_to_spans(token_preds, full_offsets)

    return spans


def group_tokens_to_spans(token_preds, offsets):
    """
    Group consecutive B/I token predictions into character-level spans.
    Handles multi-label (different labels can overlap on same tokens).
    """
    # Organize by label
    by_label = {}
    for (t_idx, label, tag, score) in token_preds:
        if label not in by_label:
            by_label[label] = []
        by_label[label].append((t_idx, tag, score))

    spans = []
    for label, tokens in by_label.items():
        tokens.sort(key=lambda x: x[0])  # sort by token index

        cur_start_tok = None
        cur_end_tok = None
        cur_scores = []

        for (t_idx, tag, score) in tokens:
            if tag == "B" or cur_start_tok is None or t_idx > cur_end_tok + 2:
                # Start new span (allow gap of 1 token for subword splits)
                if cur_start_tok is not None:
                    spans.append(finalize_span(
                        label, cur_start_tok, cur_end_tok, cur_scores, offsets
                    ))
                cur_start_tok = t_idx
                cur_end_tok = t_idx
                cur_scores = [score]
            else:
                # Continue current span
                cur_end_tok = t_idx
                cur_scores.append(score)

        if cur_start_tok is not None:
            spans.append(finalize_span(
                label, cur_start_tok, cur_end_tok, cur_scores, offsets
            ))

    return spans


def finalize_span(label, start_tok, end_tok, scores, offsets):
    """Convert token indices to character offsets."""
    char_start = offsets[start_tok][0]
    char_end = offsets[end_tok][1]
    avg_score = sum(scores) / len(scores)
    return {
        "start": int(char_start),
        "end": int(char_end),
        "label": label,
        "score": float(avg_score),
        "n_tokens": end_tok - start_tok + 1,
    }

# ============================================================
# POST-PROCESSING
# ============================================================

def post_process_spans(spans, text):
    """
    Clean and deduplicate spans based on the task's evaluation metric properties.

    Key insights from the paper:
    - Predictions >4x gold span get zero credit → avoid very long spans
    - ≥50% overlap of gold + pred ≤ 2x gold → full TP credit
    - Greedy matching → precision matters, don't flood with low-quality spans
    """
    if not spans:
        return []

    cleaned = []
    for s in spans:
        char_len = s["end"] - s["start"]
        span_text = text[s["start"]:s["end"]]

        # Remove very short spans (< 2 chars)
        if char_len < 2:
            continue

        # Remove very long spans (unlikely to be correct, and >4x gold gets 0 credit)
        # PT gold: median=63, P95=254, max=1080
        if char_len > 600:
            continue

        # Remove punctuation-only spans
        if len(re.sub(r"[^\w]", "", span_text).strip()) == 0:
            continue

        cleaned.append(s)

    # Deduplicate: for same label, if two spans overlap >50% (IoU), keep higher score
    deduped = []
    # Group by label
    by_label = {}
    for s in cleaned:
        by_label.setdefault(s["label"], []).append(s)

    for label, label_spans in by_label.items():
        label_spans.sort(key=lambda x: -x["score"])
        kept = []
        for s in label_spans:
            overlap_with_kept = False
            for k in kept:
                inter = max(0, min(s["end"], k["end"]) - max(s["start"], k["start"]))
                union = (s["end"] - s["start"]) + (k["end"] - k["start"]) - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.5:
                    overlap_with_kept = True
                    break
            if not overlap_with_kept:
                kept.append(s)
        deduped.extend(kept)

    # Sort by position
    deduped.sort(key=lambda x: (x["start"], x["end"]))
    return deduped

# ============================================================
# EVALUATION (implements the paper's Algorithm 1)
# ============================================================

def compute_overlap_score(pred, gold):
    """
    Compute S(p, g) = L(p, g) * I(p, g) as defined in the paper.
    pred and gold are dicts with 'start', 'end', 'label'.
    """
    # L(p, g): label match
    if pred["label"] != gold["label"]:
        return 0.0

    # I(p, g): span overlap
    p_start, p_end = pred["start"], pred["end"]
    g_start, g_end = gold["start"], gold["end"]

    p_len = p_end - p_start
    g_len = g_end - g_start

    if p_len <= 0 or g_len <= 0:
        return 0.0

    intersection = max(0, min(p_end, g_end) - max(p_start, g_start))

    overlap_ratio = intersection / g_len if g_len > 0 else 0  # |p ∩ g| / |g|

    if p_len <= 2 * g_len:
        if overlap_ratio >= 0.5:
            return 1.0
        else:
            return overlap_ratio  # partial credit
    elif p_len <= 4 * g_len:
        # Use |p ∩ g| / |p| instead
        return intersection / p_len if p_len > 0 else 0.0
    else:
        return 0.0


def evaluate_article(preds, golds):
    """
    Evaluate predictions for a single article using Algorithm 1 from the paper.
    Returns (TP, FP, FN).
    """
    preds = list(preds)
    golds = list(golds)
    matched = []

    # Greedy matching: find pair with max S(p,g), match them, repeat
    while preds and golds:
        best_score = -1
        best_pi = -1
        best_gi = -1

        for pi, p in enumerate(preds):
            for gi, g in enumerate(golds):
                s = compute_overlap_score(p, g)
                if s > best_score:
                    best_score = s
                    best_pi = pi
                    best_gi = gi

        if best_score <= 0:
            break

        matched.append((preds[best_pi], golds[best_gi], best_score))
        preds.pop(best_pi)
        golds.pop(best_gi)

    # Compute TP, FP, FN
    fn = len(golds)  # unmatched golds
    fp = len(preds)  # unmatched preds
    tp = 0.0

    for (p, g, s) in matched:
        tp += s
        fp += (1 - s)  # partial false positive

    return tp, fp, fn


def evaluate_all(predictions_by_doc, gold_by_doc):
    """
    Compute micro-averaged and macro-averaged F1.
    predictions_by_doc: dict {doc_id: [spans]}
    gold_by_doc: dict {doc_id: [spans]}
    """
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    # Per-technique tracking for macro F1
    technique_tp = {l: 0.0 for l in LABELS}
    technique_fp = {l: 0.0 for l in LABELS}
    technique_fn = {l: 0.0 for l in LABELS}

    all_doc_ids = set(list(gold_by_doc.keys()) + list(predictions_by_doc.keys()))

    for doc_id in all_doc_ids:
        preds = predictions_by_doc.get(doc_id, [])
        golds = gold_by_doc.get(doc_id, [])

        tp, fp, fn = evaluate_article(preds, golds)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-technique (approximate — run eval per technique per doc)
        for label in LABELS:
            label_preds = [p for p in predictions_by_doc.get(doc_id, []) if p["label"] == label]
            label_golds = [g for g in gold_by_doc.get(doc_id, []) if g["label"] == label]
            if label_preds or label_golds:
                t_tp, t_fp, t_fn = evaluate_article(label_preds, label_golds)
                technique_tp[label] += t_tp
                technique_fp[label] += t_fp
                technique_fn[label] += t_fn

    # Micro F1
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    # Per-technique metrics
    technique_details = {}
    technique_f1s = []
    for label in LABELS:
        tp = technique_tp[label]
        fp = technique_fp[label]
        fn = technique_fn[label]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        technique_f1s.append(f1)
        technique_details[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": tp + fn,  # approximate gold spans for this technique
            "pred_count": tp + fp,  # approximate pred spans for this technique
        }

    macro_f1 = np.mean(technique_f1s) if technique_f1s else 0

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_technique": technique_details,
    }

# ============================================================
# THRESHOLD SELECTION (on a source-language dev set, never on PT)
# ============================================================

def sweep_thresholds(data):
    """Evaluate every threshold in THRESHOLD_GRID on `data` (a list of
    {id, text, spans}). Returns a list of (threshold, micro_f1, macro_f1)."""
    gold_by_doc = {
        doc["id"]: [{"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}
                    for s in doc["spans"]]
        for doc in data
    }
    results = []
    for thr in THRESHOLD_GRID:
        pred_by_doc = {}
        for doc in data:
            raw = predict_document(doc["text"], threshold=thr)
            clean = post_process_spans(raw, doc["text"])
            pred_by_doc[doc["id"]] = [
                {"start": s["start"], "end": s["end"], "label": s["label"]} for s in clean
            ]
        res = evaluate_all(pred_by_doc, gold_by_doc)
        results.append((thr, res["micro_f1"], res["macro_f1"]))
    return results


def select_threshold(calib_path):
    """Select the operating threshold on a SOURCE-LANGUAGE dev set (never the PT
    gold). Returns the threshold that maximises F1-micro on that set."""
    print(f"\n=== Selecting threshold on calibration set: {calib_path} ===")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    results = sweep_thresholds(calib_data)
    best_thr, best_f1 = THRESHOLD, -1.0
    for thr, micro_f1, macro_f1 in results:
        print(f"  thr={thr:.2f}: F1-micro={micro_f1:.4f} F1-macro={macro_f1:.4f}")
        if micro_f1 > best_f1:
            best_f1, best_thr = micro_f1, thr
    print(f"  -> selected threshold = {best_thr:.2f} (F1-micro {best_f1:.4f} on calibration set)")
    return best_thr

# ============================================================
# MAIN
# ============================================================

def main():
    # ---- Determine mode: test articles or dev evaluation ----
    use_test = os.path.isdir(TEST_DIR) and len(glob.glob(os.path.join(TEST_DIR, "*.txt"))) > 0
    use_gold = os.path.isfile(GOLD_JSON)

    # ---- Select the operating threshold on the source-language dev set ----
    # The threshold is chosen here, NOT on the PT gold, and then applied unchanged
    # to both the test-article predictions and the PT gold evaluation below.
    if os.path.abspath(CALIB_JSON) == os.path.abspath(GOLD_JSON):
        sys.exit("ERROR: CALIB_JSON must not be the PT gold file. Point it at a "
                 "source-language dev set (e.g. Italian) with gold labels.")
    if os.path.isfile(CALIB_JSON):
        threshold = select_threshold(CALIB_JSON)
    else:
        print(f"WARNING: calibration file {CALIB_JSON} not found; "
              f"falling back to default threshold {THRESHOLD}.")
        threshold = THRESHOLD

    if use_test:
        print(f"\n=== Running on test articles from {TEST_DIR} ===")
        txt_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.txt")))
        print(f"Found {len(txt_files)} articles")

        all_predictions = []
        for fp in txt_files:
            doc_id = os.path.basename(fp)
            text = open(fp, encoding="utf-8", errors="ignore").read()

            raw_spans = predict_document(text, threshold=threshold)
            clean_spans = post_process_spans(raw_spans, text)

            # Format as output
            output_spans = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in clean_spans]
            all_predictions.append({
                "id": doc_id,
                "text": text,
                "spans": output_spans,
            })

            print(f"  {doc_id}: {len(clean_spans)} spans")

        # Save as JSON (same format as gold)
        os.makedirs(os.path.dirname(OUT_PREDICTIONS_JSON), exist_ok=True)
        with open(OUT_PREDICTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        print(f"\nPredictions saved to {OUT_PREDICTIONS_JSON}")

        # Also save as TSV for easy inspection
        with open(OUT_PREDICTIONS_TSV, "w", encoding="utf-8") as f:
            f.write("doc_id\tstart\tend\tlabel\n")
            for doc in all_predictions:
                for s in doc["spans"]:
                    f.write(f"{doc['id']}\t{s['start']}\t{s['end']}\t{s['label']}\n")
        print(f"TSV saved to {OUT_PREDICTIONS_TSV}")

        total_spans = sum(len(d["spans"]) for d in all_predictions)
        print(f"\nTotal: {total_spans} spans across {len(all_predictions)} articles")

    if use_gold:
        print(f"\n=== Evaluating on PT gold ({GOLD_JSON}) at threshold {threshold:.2f} ===")
        with open(GOLD_JSON, "r", encoding="utf-8") as f:
            gold_data = json.load(f)

        predictions_by_doc = {}
        gold_by_doc = {}
        total_pred_spans = 0

        for doc in gold_data:
            doc_id = doc["id"]
            text = doc["text"]

            # Gold spans
            gold_by_doc[doc_id] = [
                {"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}
                for s in doc["spans"]
            ]

            # Predict at the threshold selected on the source-language dev set
            raw_spans = predict_document(text, threshold=threshold)
            clean_spans = post_process_spans(raw_spans, text)
            predictions_by_doc[doc_id] = [
                {"start": s["start"], "end": s["end"], "label": s["label"]}
                for s in clean_spans
            ]
            total_pred_spans += len(clean_spans)

        print(f"Gold spans: {sum(len(v) for v in gold_by_doc.values())}")
        print(f"Predicted spans: {total_pred_spans}")

        # Evaluate with the paper's metric
        results = evaluate_all(predictions_by_doc, gold_by_doc)

        print(f"\n{'='*50}")
        print(f"  Threshold (from {os.path.basename(CALIB_JSON)}): {threshold:.2f}")
        print(f"  Micro Precision: {results['micro_precision']:.4f}")
        print(f"  Micro Recall:    {results['micro_recall']:.4f}")
        print(f"  Micro F1:        {results['micro_f1']:.4f}")
        print(f"  Macro F1:        {results['macro_f1']:.4f}")
        print(f"  TP: {results['total_tp']:.1f}  FP: {results['total_fp']:.1f}  FN: {results['total_fn']:.1f}")
        print(f"{'='*50}")

        # Per-technique breakdown
        print(f"\n{'='*100}")
        print(f"  {'Technique':<40} {'F1':>7} {'Prec':>7} {'Rec':>7} {'TP':>7} {'FP':>7} {'FN':>7} {'Gold#':>7} {'Pred#':>7}")
        print(f"  {'-'*96}")

        tech = results["per_technique"]
        # Sort by gold count (most frequent first)
        sorted_labels = sorted(LABELS, key=lambda l: -tech[l]["gold_count"])

        for label in sorted_labels:
            t = tech[label]
            print(f"  {label:<40} {t['f1']:>7.4f} {t['precision']:>7.4f} {t['recall']:>7.4f} "
                  f"{t['tp']:>7.1f} {t['fp']:>7.1f} {t['fn']:>7.1f} "
                  f"{t['gold_count']:>7.1f} {t['pred_count']:>7.1f}")

        print(f"  {'-'*96}")
        print(f"  {'MACRO AVERAGE':<40} {results['macro_f1']:>7.4f}")
        print(f"  {'MICRO AVERAGE':<40} {results['micro_f1']:>7.4f} {results['micro_precision']:>7.4f} {results['micro_recall']:>7.4f}")
        print(f"{'='*100}")

        # --- PT threshold curve (ANALYSIS / FIGURE ONLY) ---
        # This sweep is NOT used to select the operating threshold (that was done
        # on CALIB_JSON above). It only produces the F1-vs-threshold curve for the
        # calibration figure in the thesis.
        print(f"\n=== PT threshold curve (analysis only, not used for selection) ===")
        for thr, micro_f1, macro_f1 in sweep_thresholds(gold_data):
            print(f"  thr={thr:.2f}: F1-micro={micro_f1:.4f} F1-macro={macro_f1:.4f}")

    if not use_test and not use_gold:
        print("ERROR: No test articles found and no gold JSON found.")
        print(f"  Expected test dir: {TEST_DIR}")
        print(f"  Expected gold: {GOLD_JSON}")
        sys.exit(1)


if __name__ == "__main__":
    main()