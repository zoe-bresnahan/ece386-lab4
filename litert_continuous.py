"""This script loads a .tflite model into LiteRT and continuously takes pictures with a webcam,
printing if the picture is of a cat or a dog."""

import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys
import numpy as np


def get_litert_runner(model_path: str) -> SignatureRunner:
    """Opens a .tflite model from path and returns a LiteRT SignatureRunner that can be called for inference

    Args:
        model_path (str): Path to a .tflite model

    Returns:
        SignatureRunner: An AI-Edge LiteRT runner that can be invoked for inference."""

    interpreter = Interpreter(model_path=model_path)
    # Allocate the model in memory. Should always be called before doing inference
    interpreter.allocate_tensors()
    print(f"Allocated LiteRT with signatures {interpreter.get_signature_list()}")

    # Create callable object that runs inference based on signatures
    # 'serving_default' is default... but in production should parse from signature
    return interpreter.get_signature_runner("serving_default")


# TODO: Function to resize picture and then convert picture to numpy for model ingest
def resize(frame, size: tuple[int, int]) -> np.ndarray:
    image = cv2.resize(frame, size)

    # Convert BGR (OpenCV default) to RGB for TFLite
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to a NumPy array
    img_array = np.array(frame_rgb, dtype=np.uint8)
    print("Image shape:", img_array.shape)  # Ensure shape matches model input
    new_array = np.expand_dims(img_array, axis=0)
    print("Image shape:", new_array.shape)
    return new_array


# TODO: Function to conduct inference
def fine_pooches(numpy_array: np.ndarray, runner: SignatureRunner) -> tuple[str, float]:
    return ("sad dog", 3.14)


def main():

    # Verify arguments
    if len(sys.argv) != 2:
        print("Usage: python litert.py <model_path.tflite>")
        exit(1)

    # Create LiteRT SignatureRunner from model path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)
    # Print input and output details of runner
    print(f"Input details:\n{runner.get_input_details()}")
    print(f"Output details:\n{runner.get_output_details()}")

    # Init webcam
    webcam = cv2.VideoCapture(0)  # 0 is default camera index

    # TODO: Loop to take pictures and invoke inference. Should loop until Ctrl+C keyboard interrupt.
    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Webcam Broken!!")
                exit(1)

            # webcame took pic successfully
            np_array = resize(frame, size=(150, 150))
            print(np_array.shape)
            result, prob = fine_pooches(np_array, runner)
            print("\n")
            print(result, prob)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting gracefully.")

    # Release the camera
    webcam.release()
    print("Program complete")


# Executes when script is called by name
if __name__ == "__main__":
    main()
