# LiteRT Inference with Webcam

## Usage

Tested on Raspberry Pi 5 with USB Webcam.

After cloning,

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements-lite.txt`
4. Copy in your `.tflite` model from [Prelab](https://usafa-ece.github.io/ece386-book/b3-devboard/lab-cat-dog.html#pre-lab)
5. `python litert_continuous.py cat-dog-mnv2.tflite`

Verify your signatures are what you expect, then get to work!

## Discussion Questions

## Documentation

### People

### LLMs

Discuss the following: and add in keras model 

-Relative model sizes

-Relative performance for more vs. fewer images per run, and why

-Pipeline stalls waiting for memory

-L2 invalidations (meaning something in the L2 cache had to be overwritten)

-LLC loads and misses
