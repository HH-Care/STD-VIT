"""
STD Image Classifier using Vision Transformer (ViT-B16)

Authors:
    - Thanveer Ahamad
    - Janitha Prathapa
    - Yudara Kularathne

Description:
    This script uses a Vision Transformer (ViT-B16) model to classify penile-related STD images
    into six categories: Genital Warts, HSV, Normal, Penile Cancer, Penile Candidiasis, and Syphillis.
    It loads a pretrained ViT model, applies custom classification layers, and performs inference on a single image.

Usage:
    python std_classifier.py path/to/image.jpg

Environment Variable (Optional):
    STD_MODEL_WEIGHTS_PATH - Custom path to model weights
"""

import tensorflow as tf
from vit_keras import vit
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# -----------------------------
# Constants
# -----------------------------
IMAGE_SIZE = 224  # Input size for ViT
STD_CLASSES = [
    'Genital_warts', 'HSV', 'Normal',
    'Penile_cancer', 'Penile_candidiasis', 'Syphillis'
]
DEFAULT_WEIGHTS_PATH = '/path/to/your/weights/viT_std_model_weights.h5'  # <-- Replace this!
STD_MODEL_WEIGHTS_PATH = os.environ.get("STD_MODEL_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)


# -----------------------------
# Model Creation Function
# -----------------------------
def create_std_model():
    """
    Creates and returns the Vision Transformer (ViT-B16) model customized for STD classification.
    """
    base_model = vit.vit_b16(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    model = tf.keras.Sequential([
        base_model,
        Flatten(),  # Flatten ViT output before passing to Dense layer
        Dense(len(STD_CLASSES), activation='softmax')  # Final classification layer
    ])

    return model


# -----------------------------
# Prediction Function
# -----------------------------
def std_predict(image_path, model):
    """
    Performs prediction on the input image using the provided model.

    Args:
        image_path (str): Path to input image.
        model (tf.keras.Model): Trained STD classification model.

    Returns:
        Tuple[str, float]: Predicted class label and confidence score.
    """
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0,1]

    # Predict class probabilities
    probabilities = model.predict(img_batch)
    class_index = np.argmax(probabilities)
    class_label = STD_CLASSES[class_index]
    confidence_score = round(float(np.max(probabilities)) * 100, 2)

    # Visualize prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_label}\nConfidence: {confidence_score}%")
    plt.show()

    return class_label, confidence_score


# -----------------------------
# Main Function
# -----------------------------
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="STD Image Classifier using Vision Transformer")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    # Check if image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        exit(1)

    # Load model and weights
    model = create_std_model()
    print("Loading model weights...")
    model.load_weights(STD_MODEL_WEIGHTS_PATH)
    print("Model loaded successfully.")

    # Perform prediction
    label, confidence = std_predict(args.image_path, model)
    print(f"\nPredicted Class: {label}\nConfidence Score: {confidence}%")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
