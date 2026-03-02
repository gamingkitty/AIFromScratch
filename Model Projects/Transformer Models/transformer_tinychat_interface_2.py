import numpy as np
# import cupy as cp
import re
import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

SPECIAL_TOKENS = ["[INST]", "[/INST]", "[UNK]"]

def build_or_load_tinychat_tokenizer(
    tokenizer_path="tinychat_tokenizer.json",
    vocab_size=8000,
    min_frequency=2,
):
    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
        _validate_required_tokens(tok)
        return tok

    ds = load_dataset("starhopp3r/TinyChat", split="train")

    def text_iterator():
        for ex in ds:
            txt = ex.get("text", "")
            if txt and txt.strip():
                yield txt

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    _validate_required_tokens(tokenizer)
    tokenizer.save(tokenizer_path)
    return tokenizer


def _validate_required_tokens(tokenizer):
    missing = [t for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
    if missing:
        raise ValueError(
            f"Tokenizer is missing required special tokens: {missing}. "
            "Delete tokenizer file and retrain."
        )


def tokenize_tinychat(text, tokenizer):
    return tokenizer.encode(text).ids

def load_tinychat(
    cache_path="tinychat_indices.npz",
    tokenizer_path="tinychat_tokenizer.json",
    vocab_size=8000,
    min_frequency=2,
):
    # Fast path: load precomputed cache
    if os.path.exists(cache_path) and os.path.exists(tokenizer_path):
        cache = np.load(cache_path, allow_pickle=True)
        return list(cache["data"]), list(cache["labels"]), list(cache["vocab"])

    tokenizer = build_or_load_tinychat_tokenizer(
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    ds = load_dataset("starhopp3r/TinyChat", split="train")

    data, labels = [], []
    for ex in ds:
        text = ex.get("text", "")
        if not text or not text.strip():
            continue

        ids = tokenize_tinychat(text, tokenizer)

        # Need at least 2 tokens for next-token prediction
        if len(ids) < 2:
            continue

        arr = np.asarray(ids, dtype=np.int32)
        data.append(arr[:-1])
        labels.append(arr[1:])

    # Build id->token vocab list
    token_to_id = tokenizer.get_vocab()  # dict token -> id
    vocab_size_actual = tokenizer.get_vocab_size()
    vocab = [""] * vocab_size_actual
    for tok, idx in token_to_id.items():
        vocab[int(idx)] = tok

    np.savez_compressed(
        cache_path,
        data=np.array(data, dtype=object),
        labels=np.array(labels, dtype=object),
        vocab=np.array(vocab, dtype=object),
    )
    return data, labels, vocab


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def sample_with_temperature(probs, temperature=1.0):
    probs /= probs.sum()

    if temperature <= 0:
        return int(np.argmax(probs).item())

    # apply temperature
    scaled = np.log(probs + 1e-12) / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    new_probs = exp_scaled / exp_scaled.sum()

    idx = np.random.choice(len(probs), size=1, p=new_probs)
    return int(idx.item())


def main():
    vocab_size = len(vocab)

    print(vocab)

    language_model = Model.load("Models/tinychat_v8_13600.pkl")

    print(f"Model param num: {language_model.get_param_num()}")

    chat_history = []
    while True:
        inp = input("> ")
        if inp.lower() == "c":
            chat_history = []
            print("\nCleared chat history!\n")
            continue

        user_prompt = "[INST] " + inp + " [/INST]"
        input_tokens = tokenize_tinychat(user_prompt, tokenizer)

        print([vocab[t] for t in input_tokens])

        chat_history += input_tokens

        ai_response = []
        num = 0
        while num < 50:
            prediction = language_model.predict(np.array([chat_history]))[0][-1]
            # Can't be <unk> or [inst] so set probability to 0 (can be [/inst] though to end)
            # prediction[1] = 0
            next_token = sample_with_temperature(prediction, temperature=0.7)

            token_string = vocab[next_token]
            if token_string == "[INST]":
                break

            ai_response.append(next_token)
            chat_history.append(next_token)

            if token_string == "Ġ":
                break

            num += 1

        ai_response_string = ""
        for token in ai_response:
            ai_response_string += vocab[token]
        print(ai_response_string.replace('Ġ', ' ') + "\n")


tokenizer = build_or_load_tinychat_tokenizer()
tinychat_data, tinychat_labels, vocab = load_tinychat()
from scratch_model import *
import numpy as np

if __name__ == "__main__":
    main()
