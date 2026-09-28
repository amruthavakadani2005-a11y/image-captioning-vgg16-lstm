import os
import re
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# =========================
# Configuration
# =========================

MODEL_PATH = "caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
FEATURES_PATH = "features.pkl"
DESCRIPTIONS_PATH = "descriptions.txt"
TEST_FILE = os.path.join(
    "Flickr8k_text",
    "Flickr_8k.testImages.txt"
)

MAX_LENGTH = 30


# =========================
# Load descriptions
# =========================

def load_descriptions(filename):
    descriptions = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if len(parts) != 2:
                continue

            image_id = parts[0]
            caption = parts[1]

            image_id = image_id.split("#")[0]
            image_id = os.path.splitext(image_id)[0]

            caption = caption.lower()
            caption = re.sub(r"[^a-z ]+", "", caption)
            caption = " ".join(caption.split())

            if image_id not in descriptions:
                descriptions[image_id] = []

            descriptions[image_id].append(caption.split())

    return descriptions


# =========================
# Load test image IDs
# =========================

def load_test_ids(filename):
    test_ids = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            image_id = line.strip()

            if image_id:
                image_id = os.path.splitext(image_id)[0]
                test_ids.append(image_id)

    return test_ids


# =========================
# Generate caption
# =========================

def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        yhat = model.predict(
            [photo, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip().split()


# =========================
# Main evaluation
# =========================

print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Loading tokenizer...")

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Loading image features...")

with open(FEATURES_PATH, "rb") as file:
    features = pickle.load(file)

print("Loading captions...")

descriptions = load_descriptions(DESCRIPTIONS_PATH)

print("Loading test set...")

test_ids = load_test_ids(TEST_FILE)

print(f"Test images: {len(test_ids)}")

actual = []
predicted = []

smoother = SmoothingFunction().method4


# =========================
# Evaluate
# =========================

for i, image_id in enumerate(test_ids, start=1):

    if image_id not in features:
        continue

    if image_id not in descriptions:
        continue

    photo = features[image_id]

    caption = generate_caption(
        model,
        tokenizer,
        photo,
        MAX_LENGTH
    )

    actual.append(descriptions[image_id])
    predicted.append(caption)

    if i % 50 == 0:
        print(f"Processed {i}/{len(test_ids)} images")


# =========================
# BLEU scores
# =========================

bleu1 = corpus_bleu(
    actual,
    predicted,
    weights=(1, 0, 0, 0),
    smoothing_function=smoother
)

bleu2 = corpus_bleu(
    actual,
    predicted,
    weights=(0.5, 0.5, 0, 0),
    smoothing_function=smoother
)

bleu3 = corpus_bleu(
    actual,
    predicted,
    weights=(1/3, 1/3, 1/3, 0),
    smoothing_function=smoother
)

bleu4 = corpus_bleu(
    actual,
    predicted,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smoother
)


print("\n" + "=" * 40)
print("IMAGE CAPTIONING EVALUATION")
print("=" * 40)

print(f"Evaluated images: {len(predicted)}")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-3: {bleu3:.4f}")
print(f"BLEU-4: {bleu4:.4f}")

print("=" * 40)