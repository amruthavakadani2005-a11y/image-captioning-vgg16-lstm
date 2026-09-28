import numpy as np
import tensorflow as tf

from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


BASE_PATH = "Flickr8k_text"
DESCRIPTIONS_FILE = "descriptions.txt"
FEATURES_FILE = "features.pkl"

BATCH_SIZE = 256
EPOCHS = 5


def load_doc(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def load_set(filename):
    doc = load_doc(filename)
    dataset = set()

    for line in doc.split("\n"):
        if not line.strip():
            continue

        identifier = line.split(".")[0]
        dataset.add(identifier)

    return dataset


def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = {}

    for line in doc.split("\n"):
        tokens = line.split()

        if not tokens:
            continue

        image_id = tokens[0]
        image_desc = tokens[1:]

        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = []

            desc = "startseq " + " ".join(image_desc) + " endseq"
            descriptions[image_id].append(desc)

    return descriptions


def to_lines(descriptions):
    all_desc = []

    for key in descriptions:
        all_desc.extend(descriptions[key])

    return all_desc


def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(to_lines(descriptions))
    return tokenizer


def get_max_length(descriptions):
    return max(
        len(d.split())
        for desc_list in descriptions.values()
        for d in desc_list
    )


def load_photo_features(filename, dataset):
    all_features = load(open(filename, "rb"))
    return {k: all_features[k] for k in dataset}


def data_generator(
    descriptions,
    photos,
    tokenizer,
    max_length,
    vocab_size,
    batch_size
):
    X1, X2, y = [], [], []

    while True:
        for key, desc_list in descriptions.items():

            photo = photos[key][0]

            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]

                for i in range(1, len(seq)):

                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences(
                        [in_seq],
                        maxlen=max_length
                    )[0]

                    out_seq = to_categorical(
                        [out_seq],
                        num_classes=vocab_size
                    )[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

                    if len(X1) == batch_size:

                        yield (
                            np.array(X1, dtype="float32"),
                            np.array(X2, dtype="int32")
                        ), np.array(y, dtype="float32")

                        X1, X2, y = [], [], []


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

train_file = f"{BASE_PATH}/Flickr_8k.trainImages.txt"

train = load_set(train_file)

print("Dataset:", len(train))

train_descriptions = load_clean_descriptions(
    DESCRIPTIONS_FILE,
    train
)

print("Descriptions:", len(train_descriptions))

train_features = load_photo_features(
    FEATURES_FILE,
    train
)

print("Photos:", len(train_features))


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary Size:", vocab_size)

max_length = min(
    get_max_length(train_descriptions),
    30
)

print("Max Caption Length:", max_length)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(
        train_descriptions,
        train_features,
        tokenizer,
        max_length,
        vocab_size,
        batch_size=512
    ),
    output_signature=(
        (
            tf.TensorSpec(
                shape=(None, 4096),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, max_length),
                dtype=tf.int32
            ),
        ),
        tf.TensorSpec(
            shape=(None, vocab_size),
            dtype=tf.float32
        ),
    )
)


# --------------------------------------------------
# TRAINING STEPS
# --------------------------------------------------

total_sequences = sum(
    len(d.split()) - 1
    for v in train_descriptions.values()
    for d in v
)

steps = total_sequences // BATCH_SIZE

print("Total sequences:", total_sequences)
print("Steps per epoch:", steps)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

inputs1 = Input(shape=(4096,))

fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)


inputs2 = Input(shape=(max_length,))

se1 = Embedding(
    vocab_size,
    256,
    mask_zero=True
)(inputs2)

se2 = Dropout(0.5)(se1)

se3 = LSTM(
    256,
    use_cudnn=False
)(se2)


decoder1 = add([fe2, se3])

decoder2 = Dense(
    256,
    activation="relu"
)(decoder1)

outputs = Dense(
    vocab_size,
    activation="softmax"
)(decoder2)


model = Model(
    inputs=[inputs1, inputs2],
    outputs=outputs
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)


print(model.summary())


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "caption_model.keras",
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

model.fit(
    dataset,
    steps_per_epoch=steps,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint]
)

print("\nModel saved successfully!")
print("File: caption_model.keras")