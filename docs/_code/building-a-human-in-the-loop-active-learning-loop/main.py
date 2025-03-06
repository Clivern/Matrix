import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Labels: 0 = Safe/Good, 1 = Bad/Urgent Complaint
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

# Populated from disk (if present) or from the seed set.
training_samples: list[tuple[str, int]]

if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    training_samples = [tuple(sample) for sample in state["training_samples"]]
else:
    training_samples = list(seed_training_samples)

vectorizer: TfidfVectorizer
model: LogisticRegression

if vectorizer_path.exists() and model_path.exists():
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
else:
    vectorizer = TfidfVectorizer(stop_words="english")
    model = LogisticRegression(C=100, max_iter=1000)


def retrain_model():
    """Converts text to vector space and trains the model."""
    training_texts = [text for text, _ in training_samples]
    training_labels = [label for _, label in training_samples]

    x_train = vectorizer.fit_transform(training_texts)
    model.fit(x_train, training_labels)

    # Persist both the learned objects and the updated human-labeled data
    # so the loop can continue across script restarts.
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
    print(
        f"Human labeled as {'BAD' if human_decision == 1 else 'GOOD'}; model retrained"
    )
    return human_decision


if __name__ == "__main__":
    # If we loaded an existing vectorizer/model pair, retraining isn't required.
    # But if artifacts are missing, train once and persist them.
    if not (vectorizer_path.exists() and model_path.exists()):
        retrain_model()

    # Test 1: obvious bad review (should auto-classify)
    process_new_review("The screen is shattered and it won't turn on.")

    # Test 2: ambiguous review (human check + retrain)
    process_new_review(
        "Mixed feelings about this one.",
        human_decision=1,
    )

    # Test 3: same review again (should auto-classify after learning)
    process_new_review("Mixed feelings about this one.")
