import streamlit as st
import numpy as np
from PIL import Image
from pickle import load

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "caption_model.keras"   # or .h5
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH = 30   # must match training

# -------------------------------
# LOAD MODEL & TOKENIZER
# -------------------------------
@st.cache_resource
def load_caption_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_tokenizer():
    return load(open(TOKENIZER_PATH, "rb"))

@st.cache_resource
def load_vgg():
    base_model = VGG16()
    model = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )
    return model

model = load_caption_model()
tokenizer = load_tokenizer()
vgg_model = load_vgg()

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

# -------------------------------
# CAPTION GENERATION
# -------------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.title("🖼️ Image Caption Generator")
st.write("Upload an image and let the AI generate a caption!")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        photo = extract_features(image)
        caption = generate_caption(model, tokenizer, photo, MAX_LENGTH)

    st.success("Caption Generated!")
    st.markdown(f"### 📝 Caption:\n**{caption}**")
