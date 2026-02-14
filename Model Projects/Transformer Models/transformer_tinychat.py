import os
from collections import Counter
from datasets import load_dataset
import re
import numpy as np


def tokenize_tinychat(text):
    pattern = r"\[/?inst\]|[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]"
    return re.findall(pattern, text.lower())


def load_tinychat(cache_path="tinychat_indices.npz"):
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        return list(cache["data"]), list(cache["labels"]), list(cache["vocab"])

    ds = load_dataset("starhopp3r/TinyChat", split="train")
    conversations = [tokenize_tinychat(ex["text"]) for ex in ds if ex["text"].strip()]

    vocab = sorted(set(tok for conv in conversations for tok in conv))
    token_to_index = {t: i for i, t in enumerate(vocab)}

    data, labels = [], []
    for conv in conversations:
        if len(conv) < 2:
            continue
        idx = np.array([token_to_index[t] for t in conv], dtype=np.int32)
        data.append(idx[:-1])
        labels.append(idx[1:])  # only store indices, not one-hots

    np.savez_compressed(cache_path,
                        data=np.array(data, dtype=object),
                        labels=np.array(labels, dtype=object),
                        vocab=np.array(vocab))
    return data, labels, vocab


def load_tinychat_topk(max_vocab: int = 2500, cache_path: str = "tinychat_topk_indices_2500.npz"):
    # Fast path: load cached
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        return list(cache["data"]), list(cache["labels"]), list(cache["vocab"])

    ds = load_dataset("starhopp3r/TinyChat", split="train")

    # 1) Tokenize + count frequencies in one pass (streaming to save RAM)
    freq = Counter()
    conversations = []  # store tokenized convs; if memory is tight, chunk this
    for ex in ds:
        text = ex["text"]
        if not text or not text.strip():
            continue
        toks = tokenize_tinychat(text)
        if len(toks) < 2:
            continue
        conversations.append(toks)
        freq.update(toks)

    # 2) Build capped vocab with UNK and required specials
    specials = ["<unk>", "[inst]", "[/inst]"]  # ensure these exist
    # Remove specials from freq before selecting top-K so they don't get double-counted
    for s in specials:
        if s in freq:
            del freq[s]

    keep_n = max(0, max_vocab - len(specials))
    most_common = [t for (t, _) in freq.most_common(keep_n)]
    vocab = specials + most_common
    token_to_index = {t: i for i, t in enumerate(vocab)}
    unk_id = token_to_index["<unk>"]

    # 3) Encode each conversation to indices (no one-hot here)
    data, labels = [], []
    for toks in conversations:
        ids = np.fromiter((token_to_index.get(t, unk_id) for t in toks), dtype=np.int32)
        if ids.size < 2:
            continue
        data.append(ids[:-1])
        labels.append(ids[1:])

    # 4) Cache to disk (indices only → tiny + fast)
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


def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y


def create_block(d_model, d_feed_forward, heads, dropout_percent):
    return (
        layers.ResidualBlock(
            layers.TimeDistributedLayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, mask=model_functions.causal_mask),
            layers.Dropout(dropout_percent),
        ),
        layers.ResidualBlock(
            layers.TimeDistributedLayerNorm(),
            layers.TimeDistributedDense(d_feed_forward, model_functions.relu),
            layers.TimeDistributedDense(d_model, model_functions.linear),
            layers.Dropout(dropout_percent),
        ),
    )


def main():
    epochs = 1
    learning_rate = 0.008

    # data, labels, vocab = load_tinychat_topk()

    d_model = 320
    feed_forward_dimension = 4 * d_model
    heads = 8
    dropout_percent = 0.05
    blocks = 8

    vocab = tinychat_vocab

    vocab_size = len(vocab)

    # print("Converting arrays")
    # data = [np.array(d) for d in tinychat_data]
    # labels = [np.array(l) for l in tinychat_labels]
    # print("Finished converting arrays")
    data = tinychat_data
    labels = tinychat_labels

    print(f"Vocab Size: {vocab_size}")
    print(f"Data Size: {len(data)}")

    language_model = Model(
        model_functions.softmax_cross_entropy,
        (-1,),
        [
            layers.Embedding(d_model, vocab_size),

            *[
                layer
                for _ in range(blocks)
                for layer in create_block(d_model, feed_forward_dimension, heads, dropout_percent)
            ],

            layers.TimeDistributedLayerNorm(),

            layers.TimeDistributedDense(vocab_size, model_functions.vectorized_cross_softmax),
        ],
        accuracy_function=accuracy,
    )
    # language_model = Model.load("Models/tinychat_v2_100000")

    print(f"Param num: {language_model.get_param_num()}")

    # initial_accuracy = language_model.test(data, labels)
    # print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")

    # test_conversations = 1000
    # test_set_start = 80000
    # test_set_end = test_set_start + test_conversations
    # test_data = [np.array(data[i]) for i in range(test_set_start, test_set_end)]
    # test_labels = [np.array(to_one_hot(labels[i], vocab_size)) for i in range(test_set_start, test_set_end)]
    #
    # print(f"Testing model...")
    # test_loss, test_accuracy = language_model.test(test_data, test_labels)
    # print(f"Model has loss of {test_loss} and accuracy of {test_accuracy * 100}% on the test data set.")

    train_size = 2000
    start = 0
    while start < len(data):
        end = min(start + train_size, len(data))
        language_model.fit([np.array(data[i]) for i in range(start, end)], [np.array(to_one_hot(labels[i], vocab_size)) for i in range(start, end)], epochs, learning_rate)
        print(f"Finished training on conversations {start} to {end}")
        start += train_size
        language_model.save(f"Models/tinychat_v3_{end}")


tinychat_data, tinychat_labels, tinychat_vocab = load_tinychat_topk()
print("Loaded tinychat!")
from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()
