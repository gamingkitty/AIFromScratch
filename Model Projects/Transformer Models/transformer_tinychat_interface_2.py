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


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def sample_with_temperature(
    probs,
    temperature=1.0,
    repetition_penalty=1.0,
    recent_tokens=None,
    top_k=0,
    top_p=1.0,
):
    probs = np.asarray(probs, dtype=np.float64)

    s = probs.sum()
    probs = probs / s

    logits = np.log(probs + 1e-12)

    if repetition_penalty is not None and repetition_penalty != 1.0 and recent_tokens:
        rp = float(repetition_penalty)
        for t in set(recent_tokens):
            if 0 <= t < logits.shape[0]:
                if logits[t] > 0:
                    logits[t] /= rp
                else:
                    logits[t] *= rp

    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)

    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    new_probs = exp_logits / (exp_logits.sum() + 1e-12)

    if top_k and top_k > 0 and top_k < len(new_probs):
        k = int(top_k)
        idxs = np.argpartition(new_probs, -k)[-k:]
        mask = np.zeros_like(new_probs, dtype=bool)
        mask[idxs] = True
        new_probs = np.where(mask, new_probs, 0.0)
        new_probs = new_probs / (new_probs.sum() + 1e-12)

    if top_p is not None and top_p < 1.0:
        p = float(top_p)
        order = np.argsort(new_probs)[::-1]
        sorted_probs = new_probs[order]
        cumsum = np.cumsum(sorted_probs)
        keep = cumsum <= p

        if not np.any(keep):
            keep[0] = True
        else:
            first_over = np.argmax(~keep)
            if first_over != 0:
                keep[first_over] = True

        keep_idxs = order[keep]
        mask = np.zeros_like(new_probs, dtype=bool)
        mask[keep_idxs] = True
        new_probs = np.where(mask, new_probs, 0.0)
        new_probs = new_probs / (new_probs.sum() + 1e-12)

    if temperature is not None and temperature <= 0:
        return int(np.argmax(new_probs))

    return int(np.random.choice(len(new_probs), p=new_probs))


def main():
    vocab_size = len(vocab)

    print(vocab)

    language_model = Model.load("Models/tinychat_tinychat_tied_0005lr_0001wd_e1.pkl")

    # Temp code to support old models
    for i in range(len(language_model.layers)):
        if type(language_model.layers[i]) is layers.ResidualBlock:
            layers_to_check = language_model.layers[i].layers
            for j in range(len(layers_to_check)):
                if type(layers_to_check[j]) is layers.Attention:
                    layers_to_check[j].key_cache = []
                    layers_to_check[j].value_cache = []
                    layers_to_check[j].use_kv_cache = True
        if type(language_model.layers[i]) is layers.PositionalEncoder:
            language_model.layers[i].pos = 0

    print(f"Model param num: {language_model.get_param_num()}")

    chat_history = []
    while True:
        inp = input("> ")
        if inp.lower() == "c":
            chat_history = []
            for layer in language_model.layers:
                if type(layer) is layers.ResidualBlock:
                    for l in layer.layers:
                        if type(l) is layers.Attention:
                            l.clear_cache()
                if type(layer) is layers.PositionalEncoder:
                    layer.clear_cache()
            print("\nCleared chat history!\n")
            continue

        user_prompt = "[INST] " + inp + " [/INST]"
        input_tokens = tokenize_tinychat(user_prompt, tokenizer)

        chat_history += input_tokens

        print("Input prediction:", end="")
        for i in range(len(input_tokens) - 1):
            t = input_tokens[i]
            # Add extra dimensions for batch and time.
            inp_prediction = language_model.predict(np.array([[t]]))
            if i < len(input_tokens) - 3:
                token_string = vocab[np.argmax(inp_prediction)]
                print(token_string.replace('Ġ', ' '), end="")
        print()

        ai_response = []
        num = 0
        while num < 50:
            prediction = language_model.predict(np.array([[chat_history[-1]]]))[0][-1]

            next_token = sample_with_temperature(
                prediction,
                temperature=0.0,
                repetition_penalty=1.15,
                recent_tokens=chat_history[-64:],
                top_p=0.95,
                top_k=0,
            )

            token_string = vocab[next_token]
            if token_string == "[INST]" or token_string == "[/INST]":
                break

            if num == 0:
                token_string = token_string.lstrip('Ġ')

            print(token_string.replace('Ġ', ' '), end="")

            ai_response.append(next_token)
            chat_history.append(next_token)

            num += 1

        print()


tokenizer = build_or_load_tinychat_tokenizer()
token_to_id = tokenizer.get_vocab()
vocab_size_actual = tokenizer.get_vocab_size()
vocab = [""] * vocab_size_actual
for tok, idx in token_to_id.items():
    vocab[int(idx)] = tok
from scratch_model import *
import numpy as np

if __name__ == "__main__":
    main()
