"""This is a benchmark for running Keras model.
For a fair comparison, this code should be as similar to litert_benchmark as possible.
"""

import cv2
import numpy as np
import logging
import keras
import sys
from os.path import exists

# Threshold for binary classification
BINARY_THRESHOLD: float = 0.5

# Configure logging to print to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.info  # Alias for convenience


def webcam_to_numpy(webcam: cv2.VideoCapture, img_dims: tuple[int, int]) -> np.ndarray:
    """Capture an image from the webcam and convert it to a resized NumPy ready for inference."""
    # Capture a frame
    ret, frame = webcam.read()
    if not ret:
        logging.error("Cannot capture image with webcam")
        raise RuntimeError("Cannot capture image with webcam")

    frame = cv2.resize(frame, img_dims)
    # Convert BGR (OpenCV default) to RGB for TFLite
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to a NumPy array and reshape + add batch dimension
    return np.expand_dims(np.array(frame_rgb, dtype=np.uint8), axis=0)


def infer_dog_cat(model, img: np.ndarray) -> tuple[str, float]:
    """Run inference on a model with an image and return the prediction and probability."""
    # Run inference
    prediction = model.predict(img)[0][0]
    # Binary, so check if below threshold
    if prediction < BINARY_THRESHOLD:
        return "cat", prediction
    else:
        return "dog", prediction


if __name__ == "__main__":

    log("Application starting")

    # Make sure user provided a model file path; number of pictures is optional
    if len(sys.argv) in [2, 3]:
        model_path = sys.argv[1]
        if not exists(model_path):
            logging.error(f"File {model_path} not found.")
            exit(1)
        if len(sys.argv) == 3:
            number_pics = int(sys.argv[2])
        else:
            # Default to 10 if optional argument not provided
            number_pics = 10
    else:
        logging.error("Usage: python infer.py <model_file_path> [<number_pics>]")
        exit(1)

    model = keras.models.load_model(model_path)
    log("Model loaded")

    # Initialize the camera
    webcam = cv2.VideoCapture(0)  # 0 is the default camera index

    log(f"Running inference for {number_pics} pictures")

    # Make sure we release webcam, even if an exception occurs
    try:
        # Conduct inference until loop is complete
        for i in range(number_pics):
            img = webcam_to_numpy(webcam, (150, 150))
            prediction, probability = infer_dog_cat(model, img)
            log(prediction)
    finally:
        # Release even if exception earlier
        webcam.release()
        log("Webcam released")

    log("Program complete")
