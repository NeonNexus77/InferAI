# InferAI

InferAI is a Nyāya-inspired explainable argument analysis system. It classifies short English arguments into four pramāṇa-style labels (Pratyaksha, Anumana, Upamana, Shabda), fuses **softmax outputs with lightweight rule cues** (`classification/hybrid_reasoning.py`), estimates **ML confidence** and **adjusted hybrid confidence**, extracts **claim / premises**, surfaces **reasoning indicators** and **HTML-safe cue highlighting**, optionally returns **SHAP** over the Sentence-BERT embedding space, and emits a short **varied natural-language explanation** (`explanation_engine/explainer.py`).

## Data layout

| Path | Role |
|------|------|
| `dataset/labeling_guidelines.md` | Annotator instructions and definitions |
| `dataset/labeled_corpus.py` | Builds `master_dataset.csv` and `test_set.csv` |
| `dataset/master_dataset.csv` | Curated seed rows (`text`, `pramana_label`, `strength_label`, …) |
| `dataset/test_set.csv` | Held-out evaluation set (kept out of `nyaya_dataset.csv`) |
| `dataset/raw/nyaya_dataset.csv` | Training table (`text`, `label`) produced by `dataset_generator.py` |

## Setup

From the project root:

```bash
pip install -r requirements.txt
python dataset_generator.py
python classification/train_model.py
python reports/generate_final_evaluation_report.py
```

Training writes:

- `models/nyaya_model.pkl`, `models/label_encoder.pkl`
- `models/shap_background.npy` (SHAP `LinearExplainer` background)
- `models/evaluation/metrics.json`, `classification_report.txt`, and plot PNGs (including `confusion_matrix_test_set.png` when `dataset/test_set.csv` is present)

The Markdown report is written to `reports/final_evaluation_report.md`.

## API

```bash
uvicorn api.app:app --reload
```

POST `/analyze` with JSON:

```json
{
  "text": "Smoke is rising from the hill, therefore there must be fire.",
  "include_shap": false
}
```

Key fields include `claim`, `premises` / `evidence`, `reasoning_indicators`, `highlighted_html`, `predicted_pramana` (ML head), `hybrid_predicted_pramana`, `confidence`, `adjusted_confidence`, `reasoning_strength`, `reasoning_strength_debug`, `hybrid`, and optional `shap`.

## Streamlit UI

```bash
streamlit run frontend/app.py
```

Optional: set **`INFERAI_API_URL`** (or legacy **`NYAYAX_API_URL`**) if the API is not on `http://127.0.0.1:8000`.

## End-to-end workflow

Input text → structured extraction (claim / premises / indicators / highlights) → Sentence-BERT embeddings → logistic regression → **hybrid fusion (0.8 ML + 0.2 rules)** → composite strength → explanation → JSON (optional SHAP).
