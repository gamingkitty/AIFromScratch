from scratch_model import *
from dataset import load_or_train_tokenizer, FineWebConfig
import cupy as cp
import numpy as np


def enable_kv_cache(l_model):
    def layer_enable_kv_cache(model_layers):
        for layer in model_layers:
            if hasattr(layer, 'layers'):
                layer_enable_kv_cache(layer.layers)
            elif isinstance(layer, layers.Attention):
                layer.use_kv_cache = True
                layer.position = 0

    layer_enable_kv_cache(l_model.layers)


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

    return int(
        np.random.choice(
            len(new_probs),
            p=new_probs,
        )
    )


def main():
    language_model = Model.load(f"Models/transformer_v1_76012")
    language_model.layers[-1].set_from_embedding(language_model.layers[0])

    enable_kv_cache(language_model)

    config = FineWebConfig(
        vocab_size=12000,
        cache_directory="./fineweb_cache",
        tokens_per_shard=10_000_000,
        encoding_batch_size=256,
    )

    tokenizer = load_or_train_tokenizer(config)

    start_tokens = tokenizer.encode(input("Input tokens to continue from: "))
    print(start_tokens.tokens)

    pred = None
    for token in start_tokens.ids:
        # Add dimensions for batch and time
        pred = language_model.predict(cp.array([[token]]))

    eos_id = tokenizer.token_to_id("<eos>")

    generated_tokens = []
    while len(generated_tokens) < 512 and (len(generated_tokens) == 0 or generated_tokens[-1] != eos_id):
        next_token = sample_with_temperature(
            cp.asnumpy(pred[0][0]),
            temperature=0.7,
            repetition_penalty=1.08,
            recent_tokens=generated_tokens[-128:],
            top_p=0.95,
        )
        generated_tokens.append(next_token)
        print(tokenizer.decode([int(next_token)]), end="")
        pred = language_model.predict(cp.array([[next_token]]))

    print()
    print(len(generated_tokens))


if __name__ == "__main__":
    main()
