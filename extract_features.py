import os
import pickle
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


IMAGE_FOLDER = "Flickr8k_Dataset"
OUTPUT_FILE = "features.pkl"


def extract_features():
    base_model = VGG16()
    model = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )

    features = {}

    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images.")

    for i, filename in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGE_FOLDER, filename)

        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        features[filename] = feature

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_files)} images")

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(features, file)

    print("\nFeature extraction completed!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()