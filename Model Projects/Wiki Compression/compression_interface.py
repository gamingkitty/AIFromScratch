from scratch_model import *
from compression_main import train_or_load_tokenizer
import cupy as cp
import numpy as np
import time


def main():
    language_model = Model.load("Models/compression_model_v1_38310")
    language_model.layers[-1].set_from_embedding(language_model.layers[0])
    tokenizer = train_or_load_tokenizer(vocab_size=4096)

    prompt = input("Enter starting prompt: ")

    tokens = tokenizer.encode(prompt)

    print(f"Tokens: {tokens.tokens}")

    last_token = None

    for t in tokens.ids:
        last_token = language_model.predict(cp.array([[t]]))[0][-1]

    last_token = int(np.argmax(last_token))

    print(tokenizer.decode([last_token]), end="")

    while True:
        prediction = language_model.predict(cp.array([[last_token]]))[0][-1]
        predicted_token = int(np.argmax(prediction))

        print(tokenizer.decode([predicted_token]), end="")

        time.sleep(0.2)


if __name__ == "__main__":
    # main()
    Model.plot_csv("Loss/compression_model_v5_data", ema_span=3000)