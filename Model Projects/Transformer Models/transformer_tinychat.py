import os
from collections import Counter
from datasets import load_dataset
import re
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import time
import random
import math


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


def load_tinychat_bucketed(
    cache_path="tinychat_buckets.npz",
    tokenizer_path="tinychat_tokenizer.json",
    vocab_size=8000,
    min_frequency=2,
    min_len=2,        # minimum original token count BEFORE shift (must be >=2)
    max_len=None,     # optional cap on original token count BEFORE shift
):
    # Fast path: load precomputed cache
    if os.path.exists(cache_path) and os.path.exists(tokenizer_path):
        cache = np.load(cache_path, allow_pickle=True)

        data_buckets = list(cache["data_buckets"])
        label_buckets = list(cache["label_buckets"])

        # stored as object arrays; convert each bucket to python list
        data_buckets = [list(b) for b in data_buckets]
        label_buckets = [list(b) for b in label_buckets]

        vocab = list(cache["vocab"])
        bucket_min_seq_len = int(cache["bucket_min_seq_len"])
        return data_buckets, label_buckets, vocab, bucket_min_seq_len

    tokenizer = build_or_load_tinychat_tokenizer(
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    ds = load_dataset("starhopp3r/TinyChat", split="train")

    # We'll bucket by seq_len AFTER shift: seq_len = len(ids) - 1
    # min original len is 2 => min shifted len is 1
    bucket_min_seq_len = max(1, min_len - 1)

    # Use a dict for buckets so we don't need to know max length up front
    data_dict = {}   # seq_len -> list of arrays
    label_dict = {}  # seq_len -> list of arrays

    for ex in ds:
        text = ex.get("text", "")
        if not text or not text.strip():
            continue

        ids = tokenize_tinychat(text, tokenizer)

        # Need at least 2 tokens for next-token prediction
        if len(ids) < 2:
            continue

        # apply length filters in ORIGINAL length space if requested
        if len(ids) < min_len:
            continue
        if max_len is not None and len(ids) > max_len:
            continue

        arr = np.asarray(ids, dtype=np.int32)
        x = arr[:-1]
        y = arr[1:]
        seq_len = x.shape[0]  # == len(ids) - 1

        data_dict.setdefault(seq_len, []).append(x)
        label_dict.setdefault(seq_len, []).append(y)

    # Convert dict -> contiguous list of buckets from min..max
    if data_dict:
        max_seq_len = max(data_dict.keys())
    else:
        max_seq_len = bucket_min_seq_len  # empty dataset edge-case

    data_buckets = []
    label_buckets = []
    for L in range(bucket_min_seq_len, max_seq_len + 1):
        data_buckets.append(data_dict.get(L, []))
        label_buckets.append(label_dict.get(L, []))

    # Build id->token vocab list
    token_to_id = tokenizer.get_vocab()  # dict token -> id
    vocab_size_actual = tokenizer.get_vocab_size()
    vocab = [""] * vocab_size_actual
    for tok, idx in token_to_id.items():
        vocab[int(idx)] = tok

    np.savez_compressed(
        cache_path,
        data_buckets=np.array(data_buckets, dtype=object),
        label_buckets=np.array(label_buckets, dtype=object),
        vocab=np.array(vocab, dtype=object),
        bucket_min_seq_len=np.array(bucket_min_seq_len, dtype=np.int32),
    )

    return data_buckets, label_buckets, vocab, bucket_min_seq_len


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


def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y


def create_block(d_model, d_feed_forward, heads, dropout_percent):
    return (
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, mask=model_functions.causal_mask),
            layers.TimeDistributedDense(d_model),
            layers.Dropout(dropout_percent),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.TimeDistributedDense(d_feed_forward, model_functions.gelu),
            layers.TimeDistributedDense(d_model),
            layers.Dropout(dropout_percent),
        ),
    )


def batch_data_from_buckets(data, labels, batch_size=32, seed=4321):
    batched_data = []
    batched_labels = []

    rng = np.random.default_rng(seed)

    for i in range(len(data)):
        if len(data[i]) == 0 or len(data[i][0]) < 90 or len(data[i][0]) > 240:
            continue

        idx = rng.permutation(len(data[i]))
        data[i] = np.array(data[i])[idx]
        labels[i] = np.array(labels[i])[idx]
        batched_data.extend(np.array_split(data[i], (len(data[i]) + batch_size - 1) // batch_size))
        batched_labels.extend(np.array_split(labels[i], (len(labels[i]) + batch_size - 1) // batch_size))

    r_rng = random.Random(seed)
    idx = list(range(len(batched_data)))
    r_rng.shuffle(idx)

    batched_data = [batched_data[i] for i in idx]
    batched_labels = [batched_labels[i] for i in idx]

    return batched_data, batched_labels


def accuracy(prediction, label):
    return np.sum((np.argmax(prediction, axis=-1) == np.argmax(label, axis=-1))) / prediction.shape[1]


def lr_percent_cosine_step(step, total_steps=62538*2, warmup_steps=2000, min_percent=0.05):
    if total_steps <= 1:
        return 1.0

    step = max(0, min(int(step), total_steps - 1))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps  # 0 -> (almost) 1

    denom = total_steps - warmup_steps
    if denom <= 1:
        return 1.0

    t = (step - warmup_steps) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_percent + (1.0 - min_percent) * cosine


def main():
    learning_rate = 0.0005

    # data, labels, vocab = load_tinychat_topk()

    d_model = 384
    feed_forward_dimension = 4 * d_model
    heads = 8
    dropout_percent = 0.1
    blocks = 12

    vocab = tinychat_vocab

    vocab_size = len(vocab)

    print(f"Vocab Size: {vocab_size}")

    # language_model = Model(
    #     model_functions.vectorized_softmax_cross_entropy,
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
    #         layers.LayerNorm(),
    #
    #         layers.EmbeddingTiedOutput(vocab_size, model_functions.vectorized_cross_entropy_softmax),
    #         # layers.TimeDistributedDense(vocab_size, model_functions.vectorized_cross_entropy_softmax)
    #     ],
    #     optimizer=optimizers.AdamW,
    #     optimizer_args=(0.9, 0.999, 0.0001)  # 0.0002
    # )

    language_model = Model.load("Models/tinychat_tinychat_tied_0005lr_0001wd_20800.pkl")

    language_model.layers[-1].set_from_embedding(language_model.layers[0])

    print(f"Param num: {language_model.get_param_num()}")

    batched_data = tinychat_batched_data
    batched_labels = tinychat_batched_labels

    print(f"Number of batches: {len(batched_data)}")

    version = "tinychat_tied_0005lr_0001wd"

    csv_path = f"tinychat_{version}_data.csv"

    blocks_to_save = 20
    cur_save_num = 0
    train_size = 20
    start = 20800
    while start < len(batched_data):
        t0 = time.perf_counter()
        end = min(start + train_size, len(batched_data))

        language_model.fit(
            batched_data[start:end],
            [np.array([to_one_hot(c, vocab_size) for c in b]) for b in batched_labels[start:end]],
            1,
            learning_rate,
            is_pre_batched=True,
            batch_size=batch_size,
            accuracy_function=accuracy,
            shuffle_data=False,
            learning_rate_function=lr_percent_cosine_step,
            start_step=start
        )

        print(f"Finished training on batches {start} to {end}, taking {time.perf_counter() - t0:.4f} seconds.")
        language_model.save_csv(csv_path)

        start += train_size
        cur_save_num += 1
        model_name = f"Models/tinychat_{version}_{end}.pkl"
        if cur_save_num >= blocks_to_save:
            print(f"Saving model to {model_name}")
            language_model.save(model_name)
            cur_save_num = 0

    language_model.save(f"Models/tinychat_{version}_e1")


batch_size = 16

tinychat_tokenizer = build_or_load_tinychat_tokenizer()
# tinychat_data, tinychat_labels, tinychat_vocab = load_tinychat()
tinychat_data, tinychat_labels, tinychat_vocab, min_len = load_tinychat_bucketed()
tinychat_batched_data, tinychat_batched_labels = batch_data_from_buckets(tinychat_data, tinychat_labels, batch_size=batch_size)
print("Loaded tinychat!")
from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()
