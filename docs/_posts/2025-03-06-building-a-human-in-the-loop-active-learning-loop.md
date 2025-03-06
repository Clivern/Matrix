---
title: Building a Human in the Loop Active Learning Loop
date: 2025-03-06 00:00:00
published: true
hidden: true
sitemap: false
featured_image: https://images.unsplash.com/photo-1543599723-86e84893ebba?q=90&fm=jpg&w=1000&fit=max
excerpt: Active learning picks the examples that teach a model the most. Put a human in that loop and you spend labels where they count, not on easy cases the model already knows.
keywords: active-learning, human-in-the-loop, machine-learning, labeling, uncertainty-sampling, scikit-learn
---

![](https://images.unsplash.com/photo-1543599723-86e84893ebba?q=90&fm=jpg&w=1000&fit=max)

Active learning picks the examples that teach a model the most. Put a human in that loop and you spend labels where they count, not on easy cases the model already knows.

The cycle is simple. The model scores a new review. If it is sure, it auto-ranks. If it is not, a human labels it, that label goes back into training, and the model gets a little smarter.

```
[ New Review ]
      │
      ▼
[ Scikit-Learn Model ]
      │
      ├── High confidence? ── YES ──► [ Auto-Rank / Route ]
      │
      └── NO ──► [ Human Review ] ──► [ Update Training Set ]
                                              │
                                              └── retrain + save ──► model
```

### Setup

Install what we need with [uv](https://docs.astral.sh/uv/):

```zsh
$ uv init
$ uv add scikit-learn joblib
```

### What is the vector space?

A logistic regression model cannot read English. It only understands numbers. So before training or scoring, every review has to become a list of numbers that live in the same coordinate system. That coordinate system is the vector space.

In this demo, `TfidfVectorizer` builds that space from the words in your training set:

1. It builds a vocabulary from the labeled reviews (minus English stop words like `the` and `and`).
2. Each review becomes a sparse vector: one slot per vocabulary word.
3. The value in each slot is a TF-IDF weight: high if the word matters in that review and is not common across all reviews, low otherwise.

Two reviews that share strong words like `broken` and `shattered` land near each other in that space. A polite praise review lands somewhere else. Logistic regression then draws a boundary through the space: one side good, the other side bad.

Two calls matter:

- `fit_transform(training_texts)` learns the vocabulary and turns the training set into vectors.
- `transform([review_text])` projects a new review into that same space. It must not rebuild the vocabulary, or old and new vectors would not be comparable.

When a human labels a hard review and you retrain, `fit_transform` runs again. New words can enter the vocabulary, the axes shift a little, and the decision boundary moves with them. That is how the loop expands the space over time.

This is not the same as neural embeddings (like [sentence-transformers](/understanding-embeddings/)). TF-IDF is word counts with weights. Embeddings capture meaning even when the wording changes. For a small classifier and a clear good/bad vocabulary, TF-IDF is enough and much cheaper.

### Start with a labeled seed set

Labels: `0` = good, `1` = bad / urgent complaint. Keep each example as a `(text, label)` tuple so the pair never drifts apart. On later runs we reload whatever we already saved to disk.

```python
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

seed_training_samples = [
    ("I absolutely love this product!", 0),
    ("Amazing quality, would buy again.", 0),
    ("Fast delivery and great customer support.", 0),
    ("Perfect, exactly what I ordered.", 0),
    ("The item arrived broken and shattered.", 1),
    ("Terrible customer service, they refused my refund.", 1),
    ("Completely defective, waste of money.", 1),
    ("Worst purchase ever, do not buy.", 1),
]

base_dir = Path(__file__).resolve().parent
state_path = base_dir / "training_state.json"
vectorizer_path = base_dir / "vectorizer.joblib"
model_path = base_dir / "model.joblib"

if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    training_samples = [tuple(sample) for sample in state["training_samples"]]
else:
    training_samples = list(seed_training_samples)

if vectorizer_path.exists() and model_path.exists():
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
else:
    vectorizer = TfidfVectorizer(stop_words="english")
    model = LogisticRegression(C=100, max_iter=1000)
```

`C=100` keeps the decision boundary sharper on a small text set, so clear cases sit well above the confidence threshold.

### Retrain and save to disk

Split the tuples into texts and labels only when fitting. After every human label we refit, then write the samples, vectorizer, and model to disk so the next run picks up where we left off.

```python
def retrain_model():
    training_texts = [text for text, _ in training_samples]
    training_labels = [label for _, label in training_samples]

    x_train = vectorizer.fit_transform(training_texts)
    model.fit(x_train, training_labels)

    state_path.write_text(
        json.dumps(
            {"training_samples": training_samples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
```

### Process each new review

Use `predict_proba()` instead of a hard `predict()`. Confidence is the max class probability. If it clears a threshold, auto-label. If not, escalate to a human and retrain.

```python
def process_new_review(review_text, confidence_threshold=0.75, human_decision=None):
    review_vector = vectorizer.transform([review_text])
    probabilities = model.predict_proba(review_vector)[0]
    bad_review_prob = probabilities[1]
    confidence = max(probabilities)

    print(f"\n--- Processing: '{review_text}' ---")
    print(f"Model estimate: {bad_review_prob * 100:.1f}% chance of BAD")
    print(f"Model confidence: {confidence * 100:.1f}%")

    if confidence >= confidence_threshold:
        final_label = 1 if bad_review_prob >= 0.5 else 0
        print(f"Action: AUTO ({'BAD' if final_label == 1 else 'GOOD'})")
        return final_label

    print("Action: HUMAN CHECK REQUIRED")
    if human_decision is None:
        raise ValueError("Low confidence review needs a human label")

    training_samples.append((review_text, human_decision))
    retrain_model()
    print(f"Human labeled as {'BAD' if human_decision == 1 else 'GOOD'}; model retrained")
    return human_decision
```

### Run the loop

Train once if nothing is on disk yet, then feed a few reviews through:

```python
if not (vectorizer_path.exists() and model_path.exists()):
    retrain_model()

# Obvious bad review — should auto-classify
process_new_review("The screen is shattered and it won't turn on.")

# Ambiguous review — escalate, then learn from the human
process_new_review(
    "Mixed feelings about this one.",
    human_decision=1,
)

# Same text again — model should handle it alone now
process_new_review("Mixed feelings about this one.")
```

Run it:

```zsh
$ uv run python main.py
```

You should see something like:

```zsh
--- Processing: 'The screen is shattered and it won't turn on.' ---
Model estimate: 84.2% chance of BAD
Model confidence: 84.2%
Action: AUTO (BAD)

--- Processing: 'Mixed feelings about this one.' ---
Model estimate: 50.0% chance of BAD
Model confidence: 50.0%
Action: HUMAN CHECK REQUIRED
Human labeled as BAD; model retrained

--- Processing: 'Mixed feelings about this one.' ---
Model estimate: 97.0% chance of BAD
Model confidence: 97.0%
Action: AUTO (BAD)
```

The first ambiguous review asks for a human. The second time, the model already saw that label, so confidence rises and it passes without help. Run the script again and it loads `vectorizer.joblib`, `model.joblib`, and `training_state.json` instead of starting from scratch.

### Why this works

- **`predict_proba()`** gives `[P(good), P(bad)]` instead of a yes/no. That is what lets you measure uncertainty.
- **Confidence thresholding** (for example `0.75`) splits easy automated choices from edge cases worth a human look.
- **Continuous retraining** appends `(text, label)` tuples back into `training_samples`, so the vocabulary and decision boundary grow with real feedback.
- **`joblib.dump` / `joblib.load`** persist the vectorizer and model so the loop survives process restarts.

### Going to production

The demo stores labels in a JSON file and retrains on every human label. That is fine for a small loop. At scale:

1. **Store labels in a database** (PostgreSQL, MongoDB, whatever you already run) instead of a local JSON file.
2. **Batch retrain** on a schedule or after N new labels. Retraining after every click gets expensive once the set is large.
3. **Share the saved artifacts** (`joblib` files, or a model registry) across workers so everyone loads the same brain between retrain jobs.

The loop stays the same. Only where you keep data and how often you call `retrain_model()` changes.

[Full source code on GitHub](https://github.com/Clivern/Matrix/tree/main/docs/_code/building-a-human-in-the-loop-active-learning-loop).
