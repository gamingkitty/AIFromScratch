from scratch_model import *


def main():
    encoder_vocab_size = 100
    decoder_vocab_size = 100

    d_model = 384
    feed_forward_dimension = 4 * d_model
    heads = 8
    dropout_percent = 0.1
    blocks = 12

    translation_model = Model(
        model_functions.vectorized_softmax_cross_entropy,
        [(-1,), (-1,)],
        [
            layers.Parallel(
                [
                    # Encoder layers
                    [
                        layers.Embedding(d_model, encoder_vocab_size),

                        *[
                            layer
                            for _ in range(blocks)
                            for layer in create_block(d_model, feed_forward_dimension, heads, dropout_percent)
                        ],

                        layers.LayerNorm(),
                    ],
                    # Embedding for translated tokens
                    [
                        layers.Embedding(d_model, decoder_vocab_size),
                    ]
                ]
            ),

            layers.Reshape((-1, d_model)),

            *[
                layer
                for _ in range(blocks)
                for layer in create_block(d_model, feed_forward_dimension, heads, dropout_percent, causal=True)
            ],

            layers.LayerNorm(),

            layers.EmbeddingTiedOutput(decoder_vocab_size, model_functions.vectorized_cross_entropy_softmax),
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0001)
    )