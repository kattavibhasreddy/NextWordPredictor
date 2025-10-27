## Quick orientation

This repository is a small single-notebook project that builds a next-word predictor using Keras (TensorFlow). Key files:

- `AIML_Project.ipynb` — the main notebook. It contains code cells that:
  - load text from `corpus_file.txt` (note: one cell uses `/content/corpus_file.txt`, which indicates it was run in Colab);
  - clean and tokenize text using `Tokenizer`;
  - create sequences (default `sequence_length = 3`);
  - build an Embedding + RNN model (LSTM or GRU — the notebook shows both; the final workflow uses GRU with `units=150`, `embedding_dim=100`);
  - train with `epochs=100` and provide an `interactive_prediction()` CLI function.

- `corpus_file.txt` — training data. The notebook reads this file; keep it in the repo root and use a relative path when running locally.

## Big-picture architecture

- Single-process, notebook-driven pipeline: data load → text cleaning → tokenizer fit → sequence generation → model build (Embedding + GRU/LSTM) → training → prediction.
- State is captured as notebook globals: `tokenizer`, `sequence_length`, `model`, `total_words`. Cells depend on being executed in order; cells are not refactored into modules.

## Things an AI coding assistant should know (concrete, actionable)

1. Execution order matters: run the notebook top-to-bottom. The interactive call `interactive_prediction()` must be defined before being invoked. There are duplicate cells that call `interactive_prediction()`; re-run the defining cell if you see a NameError.

2. File paths: the notebook contains Colab-style path `/content/corpus_file.txt`. When editing or running locally on Windows, change these to a relative path like `./corpus_file.txt` or use `os.path.join(repo_root, 'corpus_file.txt')` and open with `encoding='utf-8'`.

3. Imports and runtime needs (use these exact names when ensuring environment availability):
   - pandas, re, numpy
   - tensorflow (uses `tensorflow.keras.preprocessing.text.Tokenizer`, `pad_sequences`, `Sequential`, `Embedding`, `GRU`, `LSTM`, `Dense`, `to_categorical`)

4. Concrete parameter values you can rely on when modifying code or suggesting edits:
   - `sequence_length = 3` (many cells use this value)
   - `embedding_dim = 100`, `GRU(units=150)` or `LSTM(units=150)` in alternative cells
   - `epochs=100` for training (can be reduced for quick iterations)

5. Data handling pattern: the notebook reads the full file into a single DataFrame row (`df = pd.DataFrame({'text': [text_data]})`) then creates `cleaned_text` via lowercasing and `re.sub(r'[^\w\s]', '', ...)`. Use the same transformations when writing new helper functions to preserve behavior.

6. Prediction function contract (inputs/outputs):
   - Input: `seed_text` (string)
   - Output: a single predicted token (string) or `None` if not found
   - Relies on globals: `tokenizer`, `sequence_length`, `model`.

## Recommended small fixes and conventions for PRs (safe, discoverable)

- Normalize the corpus path: replace `/content/corpus_file.txt` with `./corpus_file.txt` (or compute a path from `__file__` when moving code into modules).
- Avoid duplicate side-effecting cells (multiple calls to `interactive_prediction()`); instead keep a single interactive run cell and document how to enable it.
- If you break the notebook into modules, preserve the global names `tokenizer`, `sequence_length`, and `model` or explicitly pass them into functions to keep the notebook's logic easy to follow.

## Developer workflows (how to run & reproduce what the notebook expects)

1. Create and activate a Python venv, then install the observed runtime packages. Example (PowerShell):

```
python -m pip install --upgrade pip
python -m pip install pandas numpy tensorflow jupyter
jupyter notebook AIML_Project.ipynb
```

2. Run cells top-to-bottom. If you see errors about undefined names, re-run the preceding cells that build `tokenizer`, `model`, and `sequence` arrays.

3. There is currently no tests/CI or requirements file. Be conservative when adding new libraries; prefer only the imports observed above.

## Integration points & external assumptions

- Notebook assumes access to a CPU/GPU-backed Python environment with TensorFlow. The `/content/...` path suggests the author used Colab—be mindful of filepath and available memory when training large models locally.

## Key files to reference when making changes

- `AIML_Project.ipynb` — primary source of truth for code and execution order.
- `corpus_file.txt` — training data; never rename or relocate without updating notebook references.

If any of these sections are unclear or you'd like me to (a) merge these instructions into an existing instructions file, (b) update the notebook to normalize file paths, or (c) add a minimal `requirements.txt`, tell me which and I'll proceed.
