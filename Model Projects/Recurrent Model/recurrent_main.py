from datasets import load_dataset
import re
import numpy as np
# import cupy as cp
import os
from collections import Counter


def tokenize(text):
    tokens = re.findall(r'>[^><]*<|\n|[\w]+|[^\w\s]', text)
    return [token.lower() for token in tokens]


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    tokens = tokenize(text)

    all_tokens = set(tokens)
    all_tokens.remove(">newconversation<")
    vocab = sorted(all_tokens)
    vocab_size = len(vocab)

    token_to_index = {token: i for i, token in enumerate(vocab)}

    data = []
    labels = []

    prev_words = []
    for token in tokens:
        if token == ">newconversation<":
            data.append(np.array(prev_words[:-1]))
            labels.append(np.array([np.eye(vocab_size)[token] for token in prev_words[1:]]))
            prev_words = []
        else:
            prev_words.append(token_to_index[token])

    return data, labels, vocab


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


def load_tinychat_topk(max_vocab: int = 1000, cache_path: str = "tinychat_topk_indices.npz"):
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


def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def main():
    learning_rate = 0.007
    epochs = 1

    print("Loading Data...")
    # data, labels, vocab = load_data("Training Data/Conversations/conversations.txt")
    data, labels, vocab = tiny_data, tiny_labels, tiny_vocab
    print("Data loaded!")

    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}

    print(f"Data length: {len(data)}")
    print(f"Vocab size: {vocab_size}")

    # ai_model = Model(
    #     model_functions.cross_entropy,
    #     (-1,),
    #     [
    #         layers.Embedding(128, vocab_size, model_functions.linear),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Stack(20),
    #         layers.Loop(
    #             layers.Dense(128, model_functions.relu)
    #         ),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Stack(10),
    #         layers.Loop(
    #             layers.Dense(128, model_functions.relu)
    #         ),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Stack(5),
    #         layers.Loop(
    #             layers.Dense(512, model_functions.relu),
    #             layers.Dense(256, model_functions.relu),
    #             layers.Dense(vocab_size, model_functions.softmax)
    #         )
    #     ],
    #     accuracy_function=accuracy
    # )
    ai_model = Model(
        model_functions.cross_entropy,
        (-1,),
        [
            layers.Embedding(256, vocab_size, model_functions.linear),
            # layers.Attention(512, 512),
            # layers.Attention(256, 256),
            layers.Recurrent(128, model_functions.relu),
            # layers.Recurrent(128, model_functions.relu),
            layers.Stack(30),
            layers.Loop(
                layers.Dense(512, model_functions.relu),
                layers.Dense(256, model_functions.relu),
                layers.Dense(vocab_size, model_functions.softmax)
            )
        ],
        accuracy_function=accuracy
    )
    # ai_model = Model.load("Models/tinychat_recurrent_18000")
    print(f"Param num: {ai_model.get_param_num()}")

    # ai_model = Model.load("Models/normal_recurrent")

    # initial_accuracy = ai_model.test(data, labels)
    # print(f"Initial accuracy: {initial_accuracy * 100:.4}%")

    train_size = 500
    start = 0
    while start < len(data):
        end = min(start + train_size, len(data))
        ai_model.fit([np.array(data[i]) for i in range(start, end)], [np.array(to_one_hot(labels[i], vocab_size)) for i in range(start, end)], epochs, learning_rate)
        print(f"Finished training on conversations {start} to {end}")
        start += train_size
        ai_model.save(f"Models/tinychat_recurrent_{end}")

    final_accuracy = ai_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.4}%")

    ai_model.save("Models/tinychat_recurrent")


# tiny_data, tiny_labels, tiny_vocab = load_tinychat()
tiny_data, tiny_labels, tiny_vocab = load_tinychat_topk()
from scratch_model import *


if __name__ == "__main__":
    main()
