import numpy as np
# import cupy as cp
import re


def tokenize(text):
    pattern = r"\[/?inst\]|[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]"
    return re.findall(pattern, text.lower())


def load_vocab(cache_path: str = "tinychat_topk_indices_2500.npz"):
    cache = np.load(cache_path, allow_pickle=True)
    return list(cache["vocab"])


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def sample_with_temperature(probs, temperature=1.0):
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / probs.sum()

    if temperature <= 0:
        return int(np.argmax(probs).item())

    # apply temperature
    scaled = np.log(probs + 1e-12) / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    new_probs = exp_scaled / exp_scaled.sum()

    # cupy requires `size`
    idx = np.random.choice(len(probs), size=1, p=new_probs)
    return int(idx.item())


def main():
    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}
    known_tokens = word_to_index.keys()

    print(word_to_index)

    language_model = Model.load("Models/tinychat_v4_505000")

    print(f"Model param num: {language_model.get_param_num()}")

    chat_history = []
    while True:
        user_prompt = "[INST] " + input("> ") + " [/INST]"
        input_tokens = tokenize(user_prompt)
        not_in_vocab = False

        if not_in_vocab:
            continue

        # check for what unknown character is
        chat_history += [word_to_index[token] if token in known_tokens else 0 for token in input_tokens]

        ai_response = []
        num = 0
        while num < 50:
            # Currently just do max rather than probability distribution
            prediction = language_model.predict(np.array(chat_history))[-1]
            # Can't be <unk> or [inst] so set probability to 0 (can be [/inst] though to end)
            prediction[0] = 0
            prediction[2] = 0
            next_token = sample_with_temperature(prediction, temperature=0.7)

            token_string = index_to_word[next_token]
            if token_string == "[inst]":
                break

            ai_response.append(next_token)

            chat_history.append(next_token)

            num += 1

        ai_response_string = ""
        for token in ai_response:
            ai_response_string += index_to_word[token] + " "
        print(ai_response_string + "\n")


vocab = load_vocab()
from scratch_model import *
import numpy as np

if __name__ == "__main__":
    main()
