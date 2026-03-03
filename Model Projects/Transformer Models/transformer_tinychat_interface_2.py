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


def sample_with_temperature(
    probs,
    temperature=1.0,
    # repetition penalty (HF-style): >1 discourages recently-used tokens
    repetition_penalty=1.0,
    recent_tokens=None,
    # optional sampling filters
    top_k=0,
    top_p=1.0,
):
    probs = np.asarray(probs, dtype=np.float64)

    # Normalize early (safe even if caller forgot)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        # fallback: uniform if probs are broken
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s

    # Greedy mode (temperature <= 0) still respects repetition penalty / filters
    # (If you want true old behavior, return argmax before applying penalty/filters.)
    logits = np.log(probs + 1e-12)

    # --- repetition penalty on logits (approx HF behavior) ---
    if repetition_penalty is not None and repetition_penalty != 1.0 and recent_tokens:
        rp = float(repetition_penalty)
        # handle duplicates efficiently
        for t in set(recent_tokens):
            if 0 <= t < logits.shape[0]:
                # HF rule:
                # if logit > 0: divide by rp
                # else: multiply by rp
                if logits[t] > 0:
                    logits[t] /= rp
                else:
                    logits[t] *= rp

    # --- temperature scaling ---
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)

    # Convert to probs with softmax
    logits = logits - np.max(logits)  # stabilize
    exp_logits = np.exp(logits)
    new_probs = exp_logits / (exp_logits.sum() + 1e-12)

    # --- top_k filter ---
    if top_k and top_k > 0 and top_k < len(new_probs):
        k = int(top_k)
        idxs = np.argpartition(new_probs, -k)[-k:]
        mask = np.zeros_like(new_probs, dtype=bool)
        mask[idxs] = True
        new_probs = np.where(mask, new_probs, 0.0)
        new_probs = new_probs / (new_probs.sum() + 1e-12)

    # --- top_p (nucleus) filter ---
    if top_p is not None and top_p < 1.0:
        p = float(top_p)
        order = np.argsort(new_probs)[::-1]
        sorted_probs = new_probs[order]
        cumsum = np.cumsum(sorted_probs)
        keep = cumsum <= p
        # always keep at least 1 token
        if not np.any(keep):
            keep[0] = True
        else:
            # include the first token that crosses p
            first_over = np.argmax(~keep)
            if first_over != 0:
                keep[first_over] = True

        keep_idxs = order[keep]
        mask = np.zeros_like(new_probs, dtype=bool)
        mask[keep_idxs] = True
        new_probs = np.where(mask, new_probs, 0.0)
        new_probs = new_probs / (new_probs.sum() + 1e-12)

    # If temperature <= 0, behave like greedy after penalties/filters
    if temperature is not None and temperature <= 0:
        return int(np.argmax(new_probs))

    # Sample
    return int(np.random.choice(len(new_probs), p=new_probs))


def main():
    vocab_size = len(vocab)

    print(vocab)

    language_model = Model.load("Models/tinychat_tinychat_tied_0005lr_0001wd_37600.pkl")

    print(f"Model param num: {language_model.get_param_num()}")

    chat_history = []
    is_current_inst = True
    while True:
        inp = input("> ")
        if inp.lower() == "c":
            chat_history = []
            print("\nCleared chat history!\n")
            continue

        user_prompt = "[INST] " + inp + " [/INST]"
        input_tokens = tokenize_tinychat(user_prompt, tokenizer)

        chat_history += input_tokens

        print([vocab[t] for t in input_tokens])
        #
        # if len(chat_history) >= 64:
        #     print([vocab[t] for t in chat_history])
        #     print("Starting new conversation!")
        #     print()
        #     is_current_inst = True
        #     chat_history = [0]
        #
        # if is_current_inst:
        #     print("[INST]: ", end="")
        # else:
        #     print("[/INST]: ", end="")
        ai_response = []
        num = 0
        while num < 50:
            prediction = language_model.predict(np.array([chat_history]))[0][-1]
            # print(f"{prediction[0]}, {prediction[1]}")

            next_token = sample_with_temperature(
                prediction,
                temperature=0.7,
                repetition_penalty=1.15,
                recent_tokens=chat_history[-64:],
                top_p=0.9,
                top_k=0,
            )

            token_string = vocab[next_token]
            if token_string == "[INST]" or token_string == "[/INST]":
                # chat_history += tokenize_tinychat(f"{"[INST]" if is_current_inst else "[/INST]"}", tokenizer)
                # is_current_inst = not is_current_inst
                break

            if num == 0:
                token_string = token_string.lstrip('Ġ')

            print(token_string.replace('Ġ', ' '), end="")

            ai_response.append(next_token)
            chat_history.append(next_token)

            num += 1

        print()


tokenizer = build_or_load_tinychat_tokenizer()
tinychat_data, tinychat_labels, vocab = load_tinychat()
from scratch_model import *
import numpy as np

if __name__ == "__main__":
    main()
