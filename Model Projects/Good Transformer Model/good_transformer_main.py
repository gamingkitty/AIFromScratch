import numpy as np
import math
from scratch_model import *
from dataset import FineWebConfig, prepare_fineweb
import cupy as cp
import csv
from pathlib import Path
import time


def create_block(d_model, d_feed_forward, heads, dropout_percent):
    return (
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, mask=model_functions.causal_mask, use_rope=True, use_kv_cache=False),
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


def accuracy_function(prediction, label):
    return (cp.sum((cp.argmax(prediction, axis=-1) == label)) / prediction.shape[1]).item()


def lr_percent_cosine_step(step, total_steps=62538, warmup_steps=2000, min_percent=0.05):
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


def append_metrics_csv(file_path, step, loss, accuracy):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    file_is_empty = file_exists and path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists or file_is_empty:
            writer.writerow(["step", "loss", "accuracy"])

        writer.writerow([
            int(step),
            float(loss),
            float(accuracy),
        ])


def test_model(lm, test_iterator):
    test_iterator.reset()

    total_loss = 0
    total_accuracy = 0
    total_batches = 0
    for batch in test_iterator:
        inputs = cp.asarray(batch["inputs"])
        labels = cp.asarray(batch["targets"])

        loss, accuracy = lm.test(inputs, labels, accuracy_function=accuracy_function)

        total_loss += float(loss)
        total_accuracy += float(accuracy)
        total_batches += 1

    return total_loss / total_batches, total_accuracy / total_batches


def main():
    learning_rate = 0.0005

    d_model = 512
    feed_forward_dimension = 4 * d_model
    heads = 8
    dropout_percent = 0.1
    blocks = 14

    vocab_size = 12000

    # language_model = Model(
    #     model_functions.vectorized_softmax_cross_entropy_integer,
    #     (-1,),
    #     [
    #         layers.Embedding(d_model, vocab_size),
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
    #     ],
    #     optimizer=optimizers.AdamW,
    #     optimizer_args=(0.9, 0.999, 0.0001),
    #     dtype=cp.float32,
    #     optimizer_dtype=cp.float32,
    # )

    version = "v1"
    step = 48008

    language_model = Model.load(f"Models/transformer_{version}_{step}")
    # language_model.set_weights_dtype(cp.float32)
    # language_model.set_layer_type_dtype(layers.LayerNorm, cp.float32)
    language_model.layers[-1].set_from_embedding(language_model.layers[0])

    in_between_step = 0

    print(f"Model parameters: {language_model.get_param_num()}")

    batch_size = 6
    batch_multiplier = 4
    context_length = 1024

    test_batch_size = 2

    config = FineWebConfig(
        vocab_size=vocab_size,
        cache_directory="./fineweb_cache",
        tokens_per_shard=10_000_000,
        encoding_batch_size=256,
    )

    tokenizer, sampler, test_iterator = prepare_fineweb(
        config,
        force_retrain_tokenizer=False,
        force_rebuild_tokens=False,

        tokenizer_documents=100_000,
        cache_documents=None,

        seed=step + 1,
        test_shard_index=-1,

        test_batch_size=test_batch_size,
        context_length=context_length,
    )

    end_step = 180_000
    save_interval = 5000
    test_interval = 2000

    def learning_rate_schedule(s):
        return lr_percent_cosine_step(
            s + 1,
            total_steps=end_step + 1,
            warmup_steps=2_000,
            min_percent=0.05,
        )

    try:
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        batch_loss = 0
        batch_accuracy = 0
        while step < end_step:
            batch = sampler.sample_batch(batch_size=batch_size, context_length=context_length)

            batched_data = batch["inputs"]
            batched_labels = batch["targets"]

            cur_loss, cur_accuracy = language_model.train_batch(
                cp.asarray(batched_data),
                cp.array(batched_labels),
                accuracy_function=accuracy_function,
            )

            batch_loss += float(cur_loss)
            batch_accuracy += float(cur_accuracy)

            in_between_step += 1

            if in_between_step >= batch_multiplier:
                language_model.update_weights(learning_rate * learning_rate_schedule(step), batch_size * batch_multiplier)
                step += 1
                in_between_step = 0

                batch_loss /= batch_multiplier
                batch_accuracy /= batch_multiplier

                language_model.add_data(batch_loss, batch_accuracy, step)

                cp.cuda.Stream.null.synchronize()
                cur_time = time.time()
                print(f"Finished step {step} with loss {batch_loss:.4f} and accuracy {batch_accuracy * 100:.2f}% in {cur_time - start_time:.2f} seconds.")
                start_time = cur_time

                batch_loss = 0
                batch_accuracy = 0

                if step % save_interval == 0:
                    language_model.save(f"Models/transformer_{version}_{step}")
                    language_model.save_csv(f"Loss/transformer_{version}_train_data.csv")
                    print(f"Saved model at step {step}")

                if step % test_interval == 0:
                    print("Testing model...")
                    t0 = time.time()
                    loss, accuracy = test_model(language_model, test_iterator)
                    test_time = time.time() - t0
                    print(f"Finished testing in {test_time:.2f} seconds with loss of {loss:.4f} and accuracy of {accuracy * 100:.2f}%")
                    append_metrics_csv(f"Loss/transformer_{version}_test_data.csv", step, loss, accuracy)

    except BaseException:
        print("Error in training! Saving model.")
        language_model.save(f"Models/transformer_{version}_{step}")
        language_model.save_csv(f"Loss/transformer_{version}_train_data.csv")
        raise

    else:
        print("Finished training! Saving model.")
        language_model.save(f"Models/transformer_{version}_{step}")
        language_model.save_csv(f"Loss/transformer_{version}_train_data.csv")


if __name__ == "__main__":
    main()
    # Model.plot_csv("Loss/transformer_v1_test_data.csv", ema_span=1)
