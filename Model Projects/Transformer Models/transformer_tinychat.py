import os
from collections import Counter
from datasets import load_dataset
import re
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import time


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


# def load_tinychat_topk(max_vocab: int = 2500, cache_path: str = "tinychat_topk_indices_2500.npz"):
#     # Fast path: load cached
#     if os.path.exists(cache_path):
#         cache = np.load(cache_path, allow_pickle=True)
#         return list(cache["data"]), list(cache["labels"]), list(cache["vocab"])
#
#     ds = load_dataset("starhopp3r/TinyChat", split="train")
#
#     # 1) Tokenize + count frequencies in one pass (streaming to save RAM)
#     freq = Counter()
#     conversations = []  # store tokenized convs; if memory is tight, chunk this
#     for ex in ds:
#         text = ex["text"]
#         if not text or not text.strip():
#             continue
#         toks = tokenize_tinychat(text)
#         if len(toks) < 2:
#             continue
#         conversations.append(toks)
#         freq.update(toks)
#
#     # 2) Build capped vocab with UNK and required specials
#     specials = ["<unk>", "[inst]", "[/inst]"]  # ensure these exist
#     # Remove specials from freq before selecting top-K so they don't get double-counted
#     for s in specials:
#         if s in freq:
#             del freq[s]
#
#     keep_n = max(0, max_vocab - len(specials))
#     most_common = [t for (t, _) in freq.most_common(keep_n)]
#     vocab = specials + most_common
#     token_to_index = {t: i for i, t in enumerate(vocab)}
#     unk_id = token_to_index["<unk>"]
#
#     # 3) Encode each conversation to indices (no one-hot here)
#     data, labels = [], []
#     for toks in conversations:
#         ids = np.fromiter((token_to_index.get(t, unk_id) for t in toks), dtype=np.int32)
#         if ids.size < 2:
#             continue
#         data.append(ids[:-1])
#         labels.append(ids[1:])
#
#     # 4) Cache to disk (indices only → tiny + fast)
#     np.savez_compressed(
#         cache_path,
#         data=np.array(data, dtype=object),
#         labels=np.array(labels, dtype=object),
#         vocab=np.array(vocab, dtype=object),
#     )
#     return data, labels, vocab


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y


def create_block(d_model, d_feed_forward, heads, dropout_percent):
    return (
        layers.ResidualBlock(
            layers.TimeDistributedLayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, mask=model_functions.causal_mask),
            layers.TimeDistributedDense(d_model),
            layers.Dropout(dropout_percent),
        ),
        layers.ResidualBlock(
            layers.TimeDistributedLayerNorm(),
            layers.TimeDistributedDense(d_feed_forward, model_functions.gelu),
            layers.TimeDistributedDense(d_model),
            layers.Dropout(dropout_percent),
        ),
    )


def main():
    epochs = 1
    learning_rate = 0.005

    # data, labels, vocab = load_tinychat_topk()

    d_model = 384
    feed_forward_dimension = 4 * d_model
    heads = 6
    dropout_percent = 0.1
    blocks = 10

    vocab = tinychat_vocab

    vocab_size = len(vocab)

    data = tinychat_data
    labels = tinychat_labels

    print(f"Vocab Size: {vocab_size}")
    print(f"Data Size: {len(data)}")

    # language_model = Model(
    #     model_functions.softmax_cross_entropy,
    #     (-1,),
    #     [
    #         layers.Embedding(d_model, vocab_size),
    #         layers.PositionalEncoder(),
    #
    #         *[
    #             layer
    #             for _ in range(blocks)
    #             for layer in create_block(d_model, feed_forward_dimension, heads, dropout_percent)
    #         ],
    #
    #         layers.TimeDistributedLayerNorm(),
    #
    #         layers.TimeDistributedDense(vocab_size, model_functions.vectorized_cross_softmax),
    #     ],
    #     accuracy_function=accuracy,
    # )
    language_model = Model.load("Models/tinychat_v5_130000")

    print(f"Param num: {language_model.get_param_num()}")

    prev_models = ["Models/tinychat_v5_130000"]

    blocks_to_save = 50
    cur_save_num = 0
    train_size = 2000
    start = 130000
    while start < len(data):
        end = min(start + train_size, len(data))
        language_model.fit([np.array(data[i]) for i in range(start, end)], [np.array(to_one_hot(labels[i], vocab_size)) for i in range(start, end)], epochs, learning_rate)
        print(f"Finished training on conversations {start} to {end}")
        start += train_size
        cur_save_num += 1
        model_name = f"Models/tinychat_v5_{end}"
        if cur_save_num >= blocks_to_save:
            if len(prev_models) >= 2:
                os.remove(prev_models[0])
                prev_models.pop(0)
            prev_models.append(model_name)

            time.sleep(1)
            print(f"Saving model to {model_name}")
            language_model.save(model_name)
            cur_save_num = 0

    os.remove(prev_models[0])
    time.sleep(1)

    language_model.save(f"Models/tinychat_v5_1000000")


tinychat_tokenizer = build_or_load_tinychat_tokenizer()
tinychat_data, tinychat_labels, tinychat_vocab = load_tinychat()
print("Loaded tinychat!")
from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()
