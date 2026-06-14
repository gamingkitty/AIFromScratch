from pathlib import Path
from zipfile import ZipFile
import urllib.request
import numpy as np
import cupy as cp
from scratch_model import *
from tokenizers import ByteLevelBPETokenizer
import math
import time


ENWIK9_URL = "http://mattmahoney.net/dc/enwik9.zip"


def download_enwik9(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "enwik9.zip"
    enwik9_path = data_dir / "enwik9"

    if enwik9_path.exists():
        return enwik9_path

    if not zip_path.exists():
        print("Downloading enwik9.zip...")
        urllib.request.urlretrieve(ENWIK9_URL, zip_path)

    print("Extracting enwik9...")
    with ZipFile(zip_path, "r") as zf:
        zf.extract("enwik9", data_dir)

    return enwik9_path


def train_or_load_tokenizer(
    data_dir="data",
    vocab_size=4096,
    min_frequency=2,
):
    """
    Trains a ByteLevel BPE tokenizer on enwik9 unless one already exists.
    Stores tokenizer files in:

        data/tokenizer_vocab_{vocab_size}/
            vocab.json
            merges.txt
    """
    data_dir = Path(data_dir)
    enwik9_path = download_enwik9(data_dir)

    tokenizer_dir = data_dir / f"tokenizer_vocab_{vocab_size}"
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if vocab_path.exists() and merges_path.exists():
        print(f"Loading existing tokenizer from {tokenizer_dir}")
        tokenizer = ByteLevelBPETokenizer(
            str(vocab_path),
            str(merges_path),
        )
        return tokenizer

    print(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[str(enwik9_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
    )

    tokenizer.save_model(str(tokenizer_dir))

    print(f"Saved tokenizer to {tokenizer_dir}")
    return tokenizer


def tokenize_or_load_enwik9(
    data_dir="data",
    vocab_size=4096,
    dtype=None,
    chunk_size=4 * 1024 * 1024,  # 4 MB chunks
):
    """
    Tokenizes enwik9 in chunks to avoid huge RAM usage.

    Stores:
        data/enwik9_tokens_vocab_{vocab_size}.bin

    Loads it with np.memmap.
    """
    data_dir = Path(data_dir)
    enwik9_path = download_enwik9(data_dir)

    if dtype is None:
        dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    token_path = data_dir / f"enwik9_tokens_vocab_{vocab_size}.bin"

    if token_path.exists():
        print(f"Loading existing tokenized dataset from {token_path}")
        tokens = np.memmap(token_path, dtype=dtype, mode="r")
        print(f"Token count: {len(tokens):,}")
        return tokens

    tokenizer = train_or_load_tokenizer(
        data_dir=data_dir,
        vocab_size=vocab_size,
    )

    print("Streaming tokenization of enwik9...")

    total_bytes = 0
    total_tokens = 0

    with open(enwik9_path, "rb") as f_in, open(token_path, "wb") as f_out:
        while True:
            raw = f_in.read(chunk_size)

            if not raw:
                break

            total_bytes += len(raw)

            # latin-1 maps bytes 0..255 directly to chars 0..255
            text = raw.decode("latin-1")

            encoding = tokenizer.encode(text)
            ids = np.asarray(encoding.ids, dtype=dtype)

            ids.tofile(f_out)

            total_tokens += len(ids)

            print(
                f"\rBytes: {total_bytes:,} | "
                f"Tokens: {total_tokens:,} | "
                f"Bytes/token: {total_bytes / total_tokens:.4f}",
                end="",
            )

    print()
    print(f"Saved tokenized dataset to {token_path}")
    print(f"Original bytes: {total_bytes:,}")
    print(f"Token count:     {total_tokens:,}")
    print(f"Bytes/token:     {total_bytes / total_tokens:.4f}")

    tokens = np.memmap(token_path, dtype=dtype, mode="r")
    return tokens


def get_random_token_batch(
    tokens,
    batch_size,
    context_length,
):
    """
    Returns x, y where each row has context_length tokens.

    x: tokens[t : t + context_length]
    y: tokens[t + 1 : t + context_length + 1]

    Shapes:
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """
    max_start = len(tokens) - context_length - 1

    if max_start <= 0:
        raise ValueError("Dataset is too small for this context_length.")

    starts = np.random.randint(
        0,
        max_start,
        size=batch_size,
    )

    x = np.stack([
        tokens[s:s + context_length]
        for s in starts
    ]).astype(np.int64)

    y = np.stack([
        tokens[s + 1:s + context_length + 1]
        for s in starts
    ]).astype(np.int64)

    return x, y


def load_enwik9_token_dataset(
    data_dir="data",
    vocab_size=4096,
):
    """
    Main helper function.

    This only trains/tokenizes if the saved files do not already exist.
    """
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    tokens = tokenize_or_load_enwik9(
        data_dir=data_dir,
        vocab_size=vocab_size,
        dtype=dtype,
    )

    tokenizer = train_or_load_tokenizer(
        data_dir=data_dir,
        vocab_size=vocab_size,
    )

    return tokenizer, tokens


def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y


def lr_percent_cosine_step(step, total_steps=500000, warmup_steps=2000, min_percent=0.05):
    if total_steps <= 1:
        return 1.0

    step = max(0, min(int(step), total_steps - 1))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps  # 0 -> (almost) 1

    denom = total_steps - warmup_steps
    if denom <= 1:
        return 1.0

    t = (step - warmup_steps) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_percent + (1.0 - min_percent) * cosine


def create_block(d_model, d_feed_forward, heads):
    return (
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, mask=model_functions.causal_mask, use_rope=True, use_kv_cache=True),
            layers.TimeDistributedDense(d_model)
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.TimeDistributedDense(d_feed_forward, model_functions.gelu),
            layers.TimeDistributedDense(d_model)
        ),
    )

def accuracy(prediction, label):
    return np.sum((np.argmax(prediction, axis=-1) == label)) / prediction.shape[1]


if __name__ == "__main__":
    vocab_size = 8192
    batch_size = 4
    context_length = 2048
    learning_rate = 0.0005

    d_model = 384
    feed_forward_dimension = 4 * d_model
    heads = 6
    blocks = 10

    # language_model = Model(
    #     model_functions.vectorized_softmax_cross_entropy_integer,
    #     (-1,),
    #     [
    #         layers.Embedding(d_model, vocab_size),
    #
    #         *[
    #             layer
    #             for _ in range(blocks)
    #             for layer in create_block(d_model, feed_forward_dimension, heads)
    #         ],
    #
    #         layers.LayerNorm(),
    #
    #         layers.EmbeddingTiedOutput(vocab_size, model_functions.vectorized_cross_entropy_softmax)
    #     ],
    #     optimizer=optimizers.Adam,
    #     optimizer_args=(0.9, 0.999),
    #     dtype=cp.float32
    # )

    language_model = Model.load("Models/compression_model_v5_82267")
    language_model.layers[-1].set_from_embedding(language_model.layers[0])

    print(f"Param num: {language_model.get_param_num()}")

    tokenizer, tokens = load_enwik9_token_dataset(
        data_dir="data",
        vocab_size=vocab_size,
    )

    step = 82267

    version = "v5"

    # batched_data, batched_labels = get_random_token_batch(
    #     tokens,
    #     batch_size=1,
    #     context_length=context_length,
    # )
    #
    # batch = batched_data[0]
    #
    # pred = []
    # for t in batch:
    #     predicted_token = int(np.argmax(language_model.predict(cp.array([[t]]))[0][-1]))
    #     pred.append(predicted_token)
    #
    # print("Label:")
    # print(tokenizer.decode(batch))
    # print()
    # print("Prediction:")
    # print(tokenizer.decode(pred))

    try:
        for i in range(500000):
            batched_data, batched_labels = get_random_token_batch(
                tokens,
                batch_size=batch_size,
                context_length=context_length,
            )

            language_model.fit(
                [cp.asarray(batched_data)],
                [cp.array(batched_labels)],
                epochs=1,
                learning_rate=learning_rate,
                learning_rate_function=lr_percent_cosine_step,
                is_pre_batched=True,
                batch_size=batch_size,
                accuracy_function=accuracy,
                shuffle_data=False,
                start_step=step,
                console_updates=False,
                steps_to_update_weights=99,
                end_update_weights=False,
                data_save_file=f"Loss/compression_model_{version}_data"
            )

            if (step + 1) % 2 == 0:
                language_model.update_weights(learning_rate * lr_percent_cosine_step(step), batch_size * 2)

            step += 1

            if step % 5000 == 0:
                language_model.save(f"Models/compression_model_{version}_{step}")
                print(f"Saved model at step {step}")

    finally:
        print("Error in training! Saving model.")
        language_model.save(f"Models/compression_model_{version}_{step}")