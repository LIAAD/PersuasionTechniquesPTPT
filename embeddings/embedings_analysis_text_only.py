"""
Robustness check for the embedding-based language proximity analysis.
 
The main analysis (embedding_analysis.py) represents each document by
concatenating its text embedding with the normalised distribution of its
annotated persuasion techniques. For the Portuguese documents this draws on the
gold labels, so this script verifies that the central finding, Italian being the
nearest source language to Portuguese, does not depend on those labels.
"""

import json, os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- same config as embedding_analysis.py ----
DATA_FILES = {
    "EN": "json/train.json", "FR": "json/train_fr.json", "DE": "json/train_ge.json",
    "IT": "json/train_it.json", "PL": "json/train_po.json", "RU": "json/train_ru.json",
    "EN_dev": "json/dev.json", "FR_dev": "json/dev_fr.json", "DE_dev": "json/dev_ge.json",
    "IT_dev": "json/dev_it.json", "PL_dev": "json/dev_po.json", "RU_dev": "json/dev_ru.json",
    "PT_gold": "json/checkthat24_pt.json",
}
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 20


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_lang(lang):
    # map EN_dev -> EN so neighbours are counted per language, not per split
    return lang.replace("_dev", "")


# ---- load docs (same as original) ----
all_docs = []
for lang, path in DATA_FILES.items():
    if not os.path.isfile(path):
        print(f"  SKIP {path} (not found)")
        continue
    for doc in load_json(path):
        techniques = [s["label"] for s in doc.get("spans", [])]
        all_docs.append({
            "id": doc["id"], "lang": lang, "text": doc["text"],
            "n_spans": len(doc.get("spans", [])), "techniques": Counter(techniques),
            "is_train": "dev" not in lang and lang != "PT_gold",
            "is_pt_gold": lang == "PT_gold",
        })
print(f"Total documents: {len(all_docs)}")

# ---- text embeddings (same sampling as original) ----
print(f"Loading model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)
texts = []
for doc in all_docs:
    t = doc["text"]
    texts.append(t[:1000] + " " + t[-500:] if len(t) > 1500 else t)
embeddings = np.array(model.encode(texts, show_progress_bar=True, batch_size=32))

# ---- technique distribution features (same as original) ----
ALL_TECHNIQUES = sorted(set(tech for doc in all_docs for tech in doc["techniques"]))
tech_vectors = np.zeros((len(all_docs), len(ALL_TECHNIQUES)), dtype=np.float32)
for i, doc in enumerate(all_docs):
    total = max(doc["n_spans"], 1)
    for tech, count in doc["techniques"].items():
        tech_vectors[i, ALL_TECHNIQUES.index(tech)] = count / total

feature_sets = {
    "A: text + techniques (x5)  [original]": np.hstack([embeddings, tech_vectors * 5.0]),
    "B: text only              [no gold labels]": embeddings,
}

pt_idxs = [i for i, d in enumerate(all_docs) if d["is_pt_gold"]]
train_idxs = [i for i, d in enumerate(all_docs) if d["is_train"]]


def neighbour_distribution(feats):
    sim = cosine_similarity(feats[pt_idxs], feats[train_idxs])
    global_counts = Counter()
    dominant = Counter()
    for pi in range(len(pt_idxs)):
        top = np.argsort(sim[pi])[-TOP_K:][::-1]
        local = Counter()
        for ti in top:
            lang = base_lang(all_docs[train_idxs[ti]]["lang"])
            global_counts[lang] += 1
            local[lang] += 1
        dominant[max(local, key=local.get)] += 1
    return global_counts, dominant


# ---- run both and print side by side ----
for name, feats in feature_sets.items():
    gc, dom = neighbour_distribution(feats)
    total = sum(gc.values())
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    print("  Global neighbour share:")
    for lang, c in gc.most_common():
        print(f"    {lang:<4} {100*c/total:>5.1f}%")
    print("  Dominant language per PT doc:")
    for lang, c in dom.most_common():
        print(f"    {lang:<4} {c:>3} docs")