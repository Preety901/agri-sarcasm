# Multilingual Sarcasm Detection in Agricultural Queries Using Few-Shot Learning

B.Tech Major Project — runs fully on a MacBook (CPU / Apple Silicon). No CUDA needed.

## Pipeline
Farmer query (English / हिन्दी / Hinglish; typed, spoken or with a crop photo) → language detection →
**few-shot sarcasm classifier** → *sarcastic grievance* (priority expert desk) or *literal query*
(advisory system). The reply comes back in the language the farmer used.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt     # only for running tests
```

## Experiments (for the report)
```bash
python -m src.data                      # shows the fixed train/dev/test split (data/split.csv)
python -m src.evaluate                  # k-shot study on DEV split  -> results/summary_dev.csv, kshot_curve_dev.png
python -m src.error_analysis --k 8      # confusion matrix + wrong predictions -> results/errors.csv
python -m src.evaluate --split test     # FINAL numbers: run once, after everything is frozen
python -m src.evaluate --test-file data/fresh_test.csv   # your own fresh, unseen test set (best option)
python -m pytest -q tests               # unit tests (offline)
```
Use `--no-embed` on any command to run offline with only the TF-IDF baseline.

## Demo
```bash
streamlit run src/app.py
```

## Data split rules
* `train` pool: few-shot support examples are sampled **only** from here.
* `dev`: use for tuning (lexicon, k, prompts). `test`: touch once for final results.
* New rows added to `data/agri_sarcasm_seed.csv` get a split automatically; old assignments never change.
  Add data in batches (≥ 20 per language × label) so the stratification stays meaningful.

## Honest limitations (write these in the report)
* Seed set = 120 hand-written examples (per-language test set is ~10 queries) → results are noisy; report mean ± std.
* The incongruity lexicon was written while looking at all 120 examples, so even the test split is not fully unseen
  for that feature. For clean numbers collect a **fresh test set** and use `--test-file`.
* Languages covered: English, Hindi, Hinglish only.
* Language detection and incongruity features are lexicon-based, not learned.
* Image (CLIP), voice (Whisper) and the Ollama LLM are optional extensions, not part of the main experiment.

## Structure
```
data/agri_sarcasm_seed.csv, data/split.csv
src/data.py          fixed split + loading        src/evaluate.py       k-shot experiments
src/models.py        classifiers                   src/error_analysis.py confusion matrix + errors
src/embed.py         multilingual embeddings       src/incongruity.py    incongruity features
src/lang.py          language detection + replies  src/speech.py         Whisper speech-to-text
src/llm_fewshot.py   Ollama few-shot / advisory    src/image_incongruity.py  CLIP (optional)
src/app.py           Streamlit demo                tests/                unit tests
```
