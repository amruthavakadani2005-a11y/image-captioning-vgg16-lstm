import streamlit as st
import numpy as np
from PIL import Image
from pickle import load

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH = 30


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# Load resources
# ============================================================

@st.cache_resource
def load_caption_model():
    return load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as file:
        return load(file)


@st.cache_resource
def load_vgg():
    base_model = VGG16()

    return Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )


model = load_caption_model()
tokenizer = load_tokenizer()
vgg_model = load_vgg()


# ============================================================
# Feature extraction
# ============================================================

def extract_features(image):

    image = image.resize((224, 224))

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    feature = vgg_model.predict(
        image,
        verbose=0
    )

    return feature


# ============================================================
# Caption generation
# ============================================================

def generate_caption(
    model,
    tokenizer,
    photo,
    max_length
):

    in_text = "startseq"

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [in_text]
        )[0]

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

    return (
        in_text
        .replace("startseq", "")
        .replace("endseq", "")
        .strip()
    )


# ============================================================
# Header
# ============================================================

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Automatically generate a natural-language description "
    "for an uploaded image using a VGG16 + LSTM deep learning model."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🧠 Model")

    st.write(
        """
        **Architecture**

        VGG16 → Dense → LSTM → Softmax

        **Dataset:** Flickr8k

        **Image encoder:** VGG16

        **Caption decoder:** LSTM

        **Maximum caption length:** 30 tokens
        """
    )

    st.divider()

    st.header("📊 Evaluation")

    st.metric("BLEU-1", "0.4359")
    st.metric("BLEU-2", "0.2531")
    st.metric("BLEU-3", "0.1449")
    st.metric("BLEU-4", "0.0789")

    st.caption(
        "Evaluated on 1,000 Flickr8k test images."
    )


# ============================================================
# Upload section
# ============================================================

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    st.divider()

    if st.button(
        "✨ Generate Caption",
        type="primary"
    ):

        with st.spinner(
            "Analyzing image and generating caption..."
        ):

            try:

                photo = extract_features(image)

                caption = generate_caption(
                    model,
                    tokenizer,
                    photo,
                    MAX_LENGTH
                )

            except Exception as e:

                st.error(
                    f"An error occurred while generating the caption: {e}"
                )

                caption = ""

        if caption:

            st.success(
                "Caption generated successfully!"
            )

            st.subheader("📝 Generated Caption")

            st.info(
                f'"{caption}"'
            )

        elif caption == "":
            st.warning(
                "The model could not generate a caption "
                "for this image."
            )


# ============================================================
# How it works
# ============================================================

with st.expander("🔍 How does this work?"):

    st.markdown(
        """
        **1. Image upload**

        The user uploads an image through the Streamlit interface.

        **2. Feature extraction**

        VGG16 processes the image and extracts a
        4096-dimensional visual feature representation.

        **3. Caption generation**

        The extracted image features are combined with
        previously generated words and passed through the
        LSTM decoder.

        **4. Word prediction**

        The model predicts the next word repeatedly until
        the caption is completed.

        **5. Final caption**

        The generated sequence is displayed as the
        image description.
        """
    )


# ============================================================
# Limitations
# ============================================================

with st.expander("⚠️ Model Limitations"):

    st.markdown(
        """
        - The model was trained on the Flickr8k dataset.
        - Generated captions may occasionally contain
          incorrect objects, colors, or activities.
        - The model may perform poorly on images that are
          very different from the training data.
        - The current caption decoder uses greedy decoding.
        """
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Image Captioning using VGG16 + LSTM | "
    "Flickr8k Dataset | Built with TensorFlow & Streamlit"
)