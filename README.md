# Image Captioning using VGG16 + LSTM

An image captioning system that generates natural-language descriptions
for images using a pretrained VGG16 convolutional neural network (CNN)
and an LSTM-based decoder.

## Project Overview

Image captioning combines computer vision and natural language processing
to generate a textual description of the content of an image.

This project uses an encoder-decoder architecture:

```text
Input Image
     ↓
Pretrained VGG16 CNN
     ↓
Image Feature Extraction
     ↓
Feature Vector
     ↓
LSTM Decoder
     ↓
Generated Caption