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
 
1. Relative model sizes
The keras model was much larger than the tflite model when we ran it in Colab during the prelab. The keras model was ~26.3MB and the tflite model was ~2.39MB which is about 10x bigger. 

2. Relative performance for more vs. fewer images per run, and why
-where to find this info??

3. Pipeline stalls waiting for memory
Keras has a longer stall compared to LiteRT because it has to access memory more due to a smaller cache. The Keras model stalled cycles were 1.12e10 while LiteRT stalled cycles were 8.14e8. 

4. L2 invalidations (meaning something in the L2 cache had to be overwritten)
-keras has more (idk why)

5. LLC loads and misses
Keras read more data and had more misses compared to the LiteRT model. Keras had 4.61e8 LLC-Loads while LiteRT had 12606124 LLC-Loads. Keras had 1.75e8 LLC-Load-Misses while LiteRT had 5598791 LLC-Load-Misses. This is logical because the LiteRT model can hold more data in the higher level caches, meaning it has less to look through in the last level cache. 

## Documentation
None other than what is listed below. 

### People
EI wtih Capt Yarbrough on March 11. Also worked with my lab partner (Zoe/Raine)

### LLMs
In EI, Capt Yarbrough told us to use ChatGPT (https://chatgpt.com/c/67d0b55f-83c4-8008-913a-5c1f196d7dd2) to figure out how to end a loop when CTRL+C is pressed. 
