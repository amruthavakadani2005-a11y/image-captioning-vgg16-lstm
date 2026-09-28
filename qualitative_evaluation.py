import os
import re
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
FEATURES_PATH = "features.pkl"
DESCRIPTIONS_PATH = "descriptions.txt"
TEST_FILE = os.path.join(
    "Flickr8k_text",
    "Flickr_8k.testImages.txt"
)

MAX_LENGTH = 30
NUMBER_OF_EXAMPLES = 10


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

            descriptions[image_id].append(caption)


    return descriptions


def load_test_ids(filename):
    test_ids = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            image_id = line.strip()

            if image_id:
                image_id = os.path.splitext(image_id)[0]
                test_ids.append(image_id)

    return test_ids


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

    return in_text.replace(
        "startseq", ""
    ).replace(
        "endseq", ""
    ).strip()


print("Loading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Loading tokenizer...")

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Loading features...")

with open(FEATURES_PATH, "rb") as file:
    features = pickle.load(file)

print("Loading captions...")

descriptions = load_descriptions(
    DESCRIPTIONS_PATH
)

print("Loading test set...")

test_ids = load_test_ids(
    TEST_FILE
)


print("\n" + "=" * 60)
print("QUALITATIVE EVALUATION")
print("=" * 60)


count = 0

for image_id in test_ids:

    if image_id not in features:
        continue

    if image_id not in descriptions:
        continue

    photo = features[image_id]

    generated = generate_caption(
        model,
        tokenizer,
        photo,
        MAX_LENGTH
    )

    count += 1

    print("\n" + "-" * 60)
    print(f"IMAGE: {image_id}.jpg")

    print("\nReference captions:")

    for caption in descriptions[image_id]:
        print(f"  - {caption}")

    print("\nGenerated caption:")
    print(f"  -> {generated}")

    if count >= NUMBER_OF_EXAMPLES:
        break


print("\n" + "=" * 60)
print(f"Displayed {count} test examples.")
print("=" * 60)