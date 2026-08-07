from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class FineWebConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    split: str = "train"

    vocab_size: int = 16_000
    min_token_frequency: int = 2

    cache_directory: str = "./fineweb_cache"

    # Number of token IDs in each cached binary file.
    # With uint16, 100 million tokens occupies about 200 MB.
    tokens_per_shard: int = 100_000_000

    # Number of documents tokenized together using the Rust tokenizer backend.
    encoding_batch_size: int = 256

    seed: int = 42


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


# =============================================================================
# Paths and metadata
# =============================================================================

def _safe_cache_name(config: FineWebConfig) -> str:
    """
    Produce a stable cache name based on settings that affect tokenization.
    Context length is deliberately excluded because it does not affect the
    tokenizer or token cache.
    """
    relevant_settings = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "vocab_size": config.vocab_size,
        "min_token_frequency": config.min_token_frequency,
        "special_tokens": SPECIAL_TOKENS,
    }

    serialized = json.dumps(relevant_settings, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    dataset_label = config.dataset_name.replace("/", "_")
    config_label = config.dataset_config.replace("/", "_")

    return (
        f"{dataset_label}_{config_label}"
        f"_vocab{config.vocab_size}_{digest}"
    )


def get_cache_paths(config: FineWebConfig) -> dict[str, Path]:
    root = Path(config.cache_directory) / _safe_cache_name(config)

    return {
        "root": root,
        "tokenizer": root / "tokenizer.json",
        "metadata": root / "metadata.json",
        "shards": root / "token_shards",
    }


# =============================================================================
# Streaming FineWeb-Edu
# =============================================================================

def load_fineweb_stream(
    config: FineWebConfig,
    *,
    shuffle: bool = False,
    shuffle_buffer_size: int = 10_000,
):
    """
    Create a new FineWeb-Edu streaming dataset.

    Call this again whenever another full traversal is needed. Streaming
    iterators are consumed as they are read.
    """
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.split,
        streaming=True,
    )

    if shuffle:
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=shuffle_buffer_size,
        )

    return dataset


def iter_text_batches(
    dataset,
    batch_size: int,
) -> Iterator[list[str]]:
    """
    Yield batches of nonempty document strings.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch: list[str] = []

    for example in dataset:
        text = example.get("text")

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        batch.append(text)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# =============================================================================
# Tokenizer creation and caching
# =============================================================================

def create_byte_level_bpe_tokenizer(
    vocab_size: int,
    min_frequency: int = 2,
) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """
    Create an untrained byte-level BPE tokenizer.

    Byte-level tokenization guarantees that arbitrary UTF-8 text can be
    represented without requiring a large character vocabulary.
    """
    if vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError(
            "vocab_size is too small for the special tokens and byte alphabet"
        )

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
            byte_fallback=True,
        )
    )

    # NFKC standardizes compatible Unicode forms while preserving case.
    tokenizer.normalizer = normalizers.NFKC()

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    return tokenizer, trainer


def load_or_train_tokenizer(
    config: FineWebConfig,
    *,
    force_retrain: bool = False,
    maximum_documents: int | None = 50_000,
) -> Tokenizer:
    """
    Load a cached tokenizer, or train it on at most maximum_documents.

    Set maximum_documents=None only when you truly want to traverse
    the entire dataset.
    """
    paths = get_cache_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)

    tokenizer_path = paths["tokenizer"]

    if tokenizer_path.exists() and not force_retrain:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _validate_tokenizer(tokenizer, config)
        return tokenizer

    tokenizer, trainer = create_byte_level_bpe_tokenizer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_token_frequency,
    )

    dataset = load_fineweb_stream(config, shuffle=False)

    if maximum_documents is not None:
        dataset = dataset.take(maximum_documents)

    tokenizer.train_from_iterator(
        iter_text_batches(
            dataset,
            batch_size=config.encoding_batch_size,
        ),
        trainer=trainer,

        # This is mainly for accurate progress reporting.
        length=maximum_documents,
    )

    tokenizer.save(str(tokenizer_path), pretty=True)
    _validate_tokenizer(tokenizer, config)

    return tokenizer


def _validate_tokenizer(
    tokenizer: Tokenizer,
    config: FineWebConfig,
) -> None:
    actual_vocab_size = tokenizer.get_vocab_size()

    if actual_vocab_size > config.vocab_size:
        raise ValueError(
            f"Tokenizer has {actual_vocab_size} tokens, "
            f"but the configured maximum is {config.vocab_size}"
        )

    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"Tokenizer is missing special token {token!r}")


def get_special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    result: dict[str, int] = {}

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)

        if token_id is None:
            raise ValueError(f"Tokenizer does not contain {token!r}")

        result[token] = token_id

    return result


# =============================================================================
# Token-shard creation
# =============================================================================

def build_token_cache(
    config: FineWebConfig,
    tokenizer: Tokenizer,
    *,
    force_rebuild: bool = False,
    maximum_documents: int | None = None,
) -> dict:
    """
    Tokenize FineWeb-Edu and cache it as uint16 binary shards.

    Every document is stored as:

        document tokens, <eos>

    No padding is inserted.

    Parameters
    ----------
    maximum_documents:
        Optional debugging limit. Leave as None to process the whole selected
        FineWeb-Edu configuration.
    """
    paths = get_cache_paths(config)
    shard_directory = paths["shards"]
    metadata_path = paths["metadata"]

    if metadata_path.exists() and not force_rebuild:
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if force_rebuild and shard_directory.exists():
        for old_shard in shard_directory.glob("*.bin"):
            old_shard.unlink()

    shard_directory.mkdir(parents=True, exist_ok=True)

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            "Tokenizer vocabulary is too large for uint16 token storage"
        )

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Tokenizer does not contain <eos>")

    tokens_per_shard = config.tokens_per_shard

    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be positive")

    output_buffer = np.empty(tokens_per_shard, dtype=np.uint16)
    buffer_position = 0

    shard_lengths: list[int] = []
    total_tokens = 0
    total_documents = 0
    shard_index = 0

    def flush_shard() -> None:
        nonlocal buffer_position, shard_index

        if buffer_position == 0:
            return

        shard_path = shard_directory / f"tokens_{shard_index:05d}.bin"
        output_buffer[:buffer_position].tofile(shard_path)

        shard_lengths.append(buffer_position)

        print(
            f"Wrote {shard_path} "
            f"({buffer_position:,} tokens)"
        )

        shard_index += 1
        buffer_position = 0

    def append_tokens(token_ids: Sequence[int]) -> None:
        nonlocal buffer_position, total_tokens

        source = np.asarray(token_ids, dtype=np.uint16)
        source_position = 0

        while source_position < len(source):
            available = tokens_per_shard - buffer_position
            amount = min(available, len(source) - source_position)

            output_buffer[
                buffer_position : buffer_position + amount
            ] = source[
                source_position : source_position + amount
            ]

            buffer_position += amount
            source_position += amount
            total_tokens += amount

            if buffer_position == tokens_per_shard:
                flush_shard()

    dataset = load_fineweb_stream(config, shuffle=False)

    for text_batch in iter_text_batches(
        dataset,
        batch_size=config.encoding_batch_size,
    ):
        if maximum_documents is not None:
            documents_remaining = maximum_documents - total_documents

            if documents_remaining <= 0:
                break

            text_batch = text_batch[:documents_remaining]

        encodings = tokenizer.encode_batch(
            text_batch,
            add_special_tokens=False,
        )

        for encoding in encodings:
            # Appending EOS separately avoids constructing a second large list.
            append_tokens(encoding.ids)
            append_tokens([eos_id])
            total_documents += 1

        if total_documents % 10_000 == 0:
            print(
                f"Processed {total_documents:,} documents, "
                f"{total_tokens:,} tokens"
            )

        if (
            maximum_documents is not None
            and total_documents >= maximum_documents
        ):
            break

    flush_shard()

    if total_tokens == 0:
        raise RuntimeError("No tokens were written to the cache")

    metadata = {
        "configuration": asdict(config),
        "tokenizer_path": str(paths["tokenizer"]),
        "token_dtype": "uint16",
        "eos_token_id": eos_id,
        "total_tokens": total_tokens,
        "total_documents": total_documents,
        "shard_lengths": shard_lengths,
        "shard_files": [
            f"tokens_{index:05d}.bin"
            for index in range(len(shard_lengths))
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


# =============================================================================
# Random fixed-length sampling
# =============================================================================

class FineWebBatchSampler:
    """
    Random sampler for training shards.

    The reserved test shard is excluded from training.
    """

    def __init__(
        self,
        config,
        *,
        seed: int | None = None,
        test_shard_index: int = -1,
    ):
        paths = get_cache_paths(config)
        metadata_path = paths["metadata"]

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find cache metadata at {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.eos_token_id = int(self.metadata["eos_token_id"])
        self.rng = np.random.default_rng(
            config.seed if seed is None else seed
        )

        shard_files = self.metadata["shard_files"]
        shard_lengths = self.metadata["shard_lengths"]

        if len(shard_files) < 2:
            raise ValueError(
                "At least two shards are required to reserve one for testing."
            )

        # Convert a negative index, such as -1, into an ordinary index.
        if test_shard_index < 0:
            test_shard_index += len(shard_files)

        if not 0 <= test_shard_index < len(shard_files):
            raise IndexError(
                f"test_shard_index={test_shard_index} is invalid for "
                f"{len(shard_files)} shards"
            )

        self.test_shard_index = test_shard_index
        self.shards: list[np.memmap] = []
        self.shard_lengths: list[int] = []

        for index, (filename, length) in enumerate(
            zip(shard_files, shard_lengths)
        ):
            # Never open the test shard as part of the training sampler.
            if index == test_shard_index:
                continue

            path = paths["shards"] / filename

            shard = np.memmap(
                path,
                mode="r",
                dtype=np.uint16,
                shape=(int(length),),
            )

            self.shards.append(shard)
            self.shard_lengths.append(int(length))

        print(
            f"Training sampler loaded {len(self.shards)} shards; "
            f"reserved shard {test_shard_index} for testing."
        )

    def sample_batch(
        self,
        batch_size: int,
        context_length: int,
    ) -> dict[str, np.ndarray]:
        """
        Randomly sample a batch from training shards only.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        required_tokens = context_length + 1

        eligible_indices = [
            index
            for index, length in enumerate(self.shard_lengths)
            if length >= required_tokens
        ]

        if not eligible_indices:
            raise ValueError(
                f"No training shard contains at least "
                f"{required_tokens:,} tokens"
            )

        # Number of possible sequence windows in each shard.
        window_counts = np.asarray(
            [
                self.shard_lengths[index] - context_length
                for index in eligible_indices
            ],
            dtype=np.float64,
        )

        shard_probabilities = window_counts / window_counts.sum()

        selected_shards = self.rng.choice(
            eligible_indices,
            size=batch_size,
            replace=True,
            p=shard_probabilities,
        )

        sequences = np.empty(
            (batch_size, required_tokens),
            dtype=np.int32,
        )

        for batch_index, shard_index in enumerate(selected_shards):
            shard = self.shards[int(shard_index)]

            maximum_start = len(shard) - required_tokens

            start = int(
                self.rng.integers(
                    0,
                    maximum_start + 1,
                )
            )

            sequences[batch_index] = shard[
                start:start + required_tokens
            ]

        return {
            "inputs": sequences[:, :-1],
            "targets": sequences[:, 1:],
        }


class FineWebTestIterator:
    """
    Sequentially iterate through one reserved test shard.

    Every batch contains:
        inputs:  (batch_size, context_length)
        targets: (batch_size, context_length)

    The sequences are non-overlapping except for the one token needed to form
    shifted input/target pairs:

        sequence 1 uses tokens [0 : context_length + 1]
        sequence 2 uses tokens [context_length : 2*context_length + 1]

    This ensures every next-token transition is evaluated exactly once.
    """

    def __init__(
        self,
        config,
        *,
        batch_size: int,
        context_length: int,
        test_shard_index: int = -1,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        paths = get_cache_paths(config)
        metadata_path = paths["metadata"]

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find cache metadata at {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        shard_files = metadata["shard_files"]
        shard_lengths = metadata["shard_lengths"]

        if not shard_files:
            raise ValueError("The token cache contains no shards")

        if test_shard_index < 0:
            test_shard_index += len(shard_files)

        if not 0 <= test_shard_index < len(shard_files):
            raise IndexError(
                f"test_shard_index={test_shard_index} is invalid for "
                f"{len(shard_files)} shards"
            )

        self.batch_size = batch_size
        self.context_length = context_length
        self.drop_last = drop_last
        self.test_shard_index = test_shard_index

        self.shard_filename = shard_files[test_shard_index]
        self.shard_length = int(shard_lengths[test_shard_index])

        shard_path = paths["shards"] / self.shard_filename

        self.shard = np.memmap(
            shard_path,
            mode="r",
            dtype=np.uint16,
            shape=(self.shard_length,),
        )

        # Current start token for the next context.
        self.position = 0

        # A context of length L needs L + 1 stored tokens.
        self.sequence_count = max(
            0,
            (self.shard_length - 1) // self.context_length,
        )

        if self.drop_last:
            self.batch_count = self.sequence_count // self.batch_size
        else:
            self.batch_count = (
                self.sequence_count + self.batch_size - 1
            ) // self.batch_size

        print(
            f"Test shard: {self.shard_filename}\n"
            f"Tokens: {self.shard_length:,}\n"
            f"Sequences: {self.sequence_count:,}\n"
            f"Batches: {self.batch_count:,}"
        )

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        """
        Return the next sequential batch.

        Raises StopIteration after the whole test shard has been traversed.
        """
        maximum_sequence_count = (
            self.shard_length - 1 - self.position
        ) // self.context_length

        if maximum_sequence_count <= 0:
            raise StopIteration

        current_batch_size = min(
            self.batch_size,
            maximum_sequence_count,
        )

        if self.drop_last and current_batch_size < self.batch_size:
            raise StopIteration

        sequences = np.empty(
            (
                current_batch_size,
                self.context_length + 1,
            ),
            dtype=np.int32,
        )

        for row in range(current_batch_size):
            start = self.position
            end = start + self.context_length + 1

            sequences[row] = self.shard[start:end]

            # Advance by context_length, not context_length + 1.
            # The shared token is needed for the next input/target transition.
            self.position += self.context_length

        return {
            "inputs": sequences[:, :-1],
            "targets": sequences[:, 1:],
        }

    def next(self) -> dict[str, np.ndarray]:
        """
        Allows test_iterator.next() in addition to next(test_iterator).
        """
        return self.__next__()

    def reset(self) -> None:
        """
        Restart iteration from the beginning of the test shard.
        """
        self.position = 0

    def batches_remaining(self) -> int:
        """
        Return the number of complete batches remaining.
        """
        remaining_sequences = max(
            0,
            (self.shard_length - 1 - self.position)
            // self.context_length,
        )

        if self.drop_last:
            return remaining_sequences // self.batch_size

        return (
            remaining_sequences + self.batch_size - 1
        ) // self.batch_size

    def progress(self) -> float:
        """
        Return traversal progress from 0.0 to 1.0.
        """
        usable_tokens = self.sequence_count * self.context_length

        if usable_tokens == 0:
            return 1.0

        return min(1.0, self.position / usable_tokens)


def create_train_and_test_loaders(
    config,
    *,
    train_seed: int = 1,
    test_shard_index: int = -1,
    test_batch_size: int = 4,
    context_length: int = 1024,
    drop_last_test_batch: bool = True,
):
    """
    Create a random training sampler and sequential test iterator that use
    non-overlapping shards.
    """
    train_sampler = FineWebBatchSampler(
        config,
        seed=train_seed,
        test_shard_index=test_shard_index,
    )

    test_iterator = FineWebTestIterator(
        config,
        batch_size=test_batch_size,
        context_length=context_length,
        test_shard_index=test_shard_index,
        drop_last=drop_last_test_batch,
    )

    return train_sampler, test_iterator


def create_document_causal_mask(
    segment_ids: np.ndarray,
) -> np.ndarray:
    """
    Build a block-diagonal causal attention mask.

    Returns
    -------
    mask:
        Boolean array of shape:

            (batch_size, context_length, context_length)

        mask[b, query, key] is True when the query may attend to that key.

    Warning:
        This dense representation uses O(batch * context_length^2) memory.
        For long contexts, generate the same condition inside your attention
        kernel rather than storing the full mask.
    """
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape (batch, sequence)")

    _, context_length = segment_ids.shape

    same_document = (
        segment_ids[:, :, None]
        == segment_ids[:, None, :]
    )

    causal = np.tril(
        np.ones(
            (context_length, context_length),
            dtype=bool,
        )
    )

    return same_document & causal[None, :, :]


# =============================================================================
# Convenience setup function
# =============================================================================

def prepare_fineweb(
    config: FineWebConfig,
    *,
    force_retrain_tokenizer: bool = False,
    force_rebuild_tokens: bool = False,
    tokenizer_documents: int | None = 50_000,
    cache_documents: int | None = None,
    seed: int | None = None,
    test_shard_index: int = -1,
    test_batch_size: int = 4,
    context_length: int = 1024,
    drop_last_test_batch: bool = True,
) -> tuple[
    Tokenizer,
    FineWebBatchSampler,
    FineWebTestIterator,
]:
    """
    Prepare FineWeb-Edu for training and testing.

    This function:

    1. Loads or trains the tokenizer.
    2. Loads or creates the token cache.
    3. Reserves one cached shard for testing.
    4. Creates a random sampler using every other shard.
    5. Creates a sequential iterator over the test shard.

    Returns
    -------
    tokenizer:
        The cached or newly trained tokenizer.

    train_sampler:
        Random sampler that excludes the reserved test shard.

    test_iterator:
        Sequential iterator over the reserved test shard.
        Supports both:
            next(test_iterator)
        and:
            test_iterator.next()
    """

    tokenizer = load_or_train_tokenizer(
        config,
        force_retrain=force_retrain_tokenizer,
        maximum_documents=tokenizer_documents,
    )

    build_token_cache(
        config,
        tokenizer,
        force_rebuild=force_rebuild_tokens,
        maximum_documents=cache_documents,
    )

    train_sampler = FineWebBatchSampler(
        config,
        seed=seed,
        test_shard_index=test_shard_index,
    )

    test_iterator = FineWebTestIterator(
        config,
        batch_size=test_batch_size,
        context_length=context_length,
        test_shard_index=test_shard_index,
        drop_last=drop_last_test_batch,
    )

    return tokenizer, train_sampler, test_iterator