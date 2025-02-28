import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys


def get_litert_runner(model_path: str) -> SignatureRunner:
    """Opens a .tflite model from path,
    returns a LiteRT SignatureRunner
    that can be called for inference"""
    interpreter = Interpreter(model_path=model_path)
    # Allocate the model in memory. Should always be called before doing inference
    interpreter.allocate_tensors()
    print(f"Allocated LiteRT with {interpreter.get_signature_list()}")

    # Create callable object that runs inference based on signatures
    # 'serving_default' is default... but need to compare against signature
    return interpreter.get_signature_runner("serving_default")


# TODO: Convert picture to numpy for model

# TODO: Inference


def main():
    # Init webcam
    webcam = cv2.VideoCapture(0)  # 0 is default camera index

    # Allocate model based on path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)
    # Print input and output details of runner
    print(f"Input details:\n{runner.get_input_details()}")
    print(f"Output details:\n{runner.get_output_details()}")

    # TODO: Loop to take pictures

    # Release the camera
    webcam.release()
    print("test")


# Executes when script is called by name
if __name__ == "__main__":
    main()
