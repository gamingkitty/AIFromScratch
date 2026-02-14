from keras.datasets import mnist
from scratch_model import *
import numpy as np


def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    # train_images_f = train_images_f.reshape((-1, 28*28))
    # test_images_f = test_images_f.reshape((-1, 28*28))

    return train_images_f, train_labels_f, test_images_f, test_labels_f


def main():
    save_as = "Models/dense_test"

    train_images, train_labels, test_images, test_labels = load_data()

    ohe_train_labels = np.eye(10)[train_labels]
    ohe_test_labels = np.eye(10)[test_labels]

    # ai_model = model.Model.load(save_as)
    # ai_model = Model(
    #     model_functions.cross_entropy,
    #     (28, 28),
    #     [
    #         layers.Dense(64, model_functions.relu),
    #         layers.Dense(10, model_functions.linear)
    #     ]
    # )
    ai_model = Model(
        model_functions.cross_entropy,
        (1, 28, 28),
        [
            layers.Convolution(32, (3, 3), model_functions.relu),
            layers.MaxPooling((2, 2), 2),

            layers.Convolution(64, (3, 3), model_functions.relu),
            # layers.MaxPooling((2, 2), 2),
            layers.Dense(128, model_functions.relu),
            layers.Dense(64, model_functions.relu),
            # layers.Dropout(0.5),

            layers.Dense(10, model_functions.softmax)
        ]
    )

    # accuracy = ai_model.test(test_images, ohe_test_labels)
    # print(f"Initial model accuracy is {accuracy * 100}%")
    # print()

    ai_model.fit(train_images, ohe_train_labels, 10, 0.01)

    print()
    # accuracy = ai_model.test(test_images, ohe_test_labels)
    # print(f"Final model accuracy is {accuracy * 100}%")
    # print()

    print("Saving model...")
    ai_model.save(save_as)
    print(f"Saved model as {save_as}.pkl")


if __name__ == "__main__":
    # main()
    # values = np.array([
    #     [1, 2],
    #     [4, 5],
    #     [6, 7],
    # ])
    #
    # attention = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5],
    #     [5, 6, 7]
    # ])

    # test = np.array([1, 2, 3])
    #
    # print(np.tensordot(attention, values, axes=[1,0]))

    # keys = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6]
    # ])
    #
    # queries = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5],
    # ])
    #
    # raw_attention_scores = np.tensordot(queries, keys, axes=[1, 1])
    #
    # # Softmax attention scores along key axis
    # e_xs = np.exp(raw_attention_scores - np.max(raw_attention_scores, axis=1, keepdims=True))
    # attention_scores = e_xs / np.sum(e_xs, axis=1, keepdims=True)
    #
    # print(raw_attention_scores)
    # print(attention_scores)

    # test = np.array([
    #     [1, 2, 3],
    #     [1, 3, 4]
    # ])
    # test_da_dz = np.array([
    #     [
    #         [3, 4, 5],
    #         [5, 6, 7],
    #         [6, 7, 8]
    #     ],
    #     [
    #         [3, 4, 6],
    #         [2, 6, 1],
    #         [6, 3, 8]
    #     ]
    # ])

    # print(np.dot(test_dc_da[1], test_da_dz[1]))
    # print(np.einsum('ij,ijk->ik', test_dc_da, test_da_dz))

    # print(np.einsum('ij,ijk->ik', test, test_da_dz))
    # print(np.dot(test[1], test_da_dz[1]))

    # values = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6]
    # ])
    #
    # dc_da = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5]
    # ])
    #
    # print(np.dot(dc_da, values.T))

    g = np.array([
        [1, 2, 3],
        [3, 4, 5]
    ])

    quotient = np.array([
        [3, 4, 5],
        [6, 7, 8]
    ])

    epsilon_std = np.array([2, 3])

    # print(dc_da @ values.T)

    # print(test_da_dz[:, :, np.newaxis])


    # 2 tokens, 3 length value
    # values = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5],
    # ])


    # print(np.tensordot(attentions.T, values, axes=[1, 0]))

    # print(np.tensordot(attentions, dc_da, axes=[1, 0]))

    # dr_dq = np.array([
    #     [1, 2],
    #     [3, 4],
    # ])
    #
    # dc_dr = np.array([
    #     [5, 6],
    #     [5, 8]
    # ])

    # token = np.array([
    #     [1, 2],
    #     [1, 3]
    # ])
    #
    # dc_dq = np.array([
    #     [4, 5, 6],
    #     [4, 5, 6]
    # ])
    #
    # print(np.sum(token[:, :, np.newaxis] * dc_dq[:, np.newaxis], axis=0))


    test_input = np.array([
        [1, 2],
        [4, 5],
        [2, 3],
    ])

    dc_da = np.array([
        [
            [1, 2],
            [4, 5],
            [2, 3],
        ],
        [
            [1, 3],
            [2, 5],
            [2, 4],
        ]
    ])



    test_queries = np.array([
        [
            [2, 3],
            [4, 5],
            [6, 7]
        ],
        [
            [1, 3],
            [4, 5],
            [6, 7]
        ]
    ])

    test_keys = np.array([
        [
            [2, 3],
            [2, 4],
            [5, 7]
        ],
        [
            [1, 3],
            [3, 5],
            [6, 7]
        ]
    ])

    test_key_weights = np.array([
        [
            [1, 2],
            [3, 4]
        ],
        [
            [3, 4],
            [5, 6]
        ],
    ])

    test_mask = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    attentions = np.einsum('htv,hsv->hts', test_queries, test_keys)
    # print(attentions)
    # # print(attentions * test_mask)
    # e_xs = np.exp(attentions[1] - np.max(attentions[1], axis=1, keepdims=True))
    # softmax = e_xs / np.sum(e_xs, axis=1, keepdims=True)
    # print(softmax)

    # print(np.tensordot(attentions[0], test_queries[0], axes=[1, 0]))

    # dc_dv = np.einsum('hts,tv->hsv', attentions, dc_da)

    h, n, m = attentions.shape
    idx = np.arange(m)
    diagonals = np.zeros((h, n, m, m), dtype=attentions.dtype)
    diagonals[:, :, idx, idx] = attentions

    dattention_draw = diagonals - (attentions[:, :, :, np.newaxis] * attentions[:, :, np.newaxis, :])
    # print(dattention_draw)

    dc_draw = np.sum((attentions[:, :, :, np.newaxis] * dattention_draw), axis=2)

    # print(np.dot(dc_draw[1].T, test_queries[1]))
    # print(np.einsum('hts,htk->hsk', dc_draw, test_queries))

    dc_draw_test = attentions * (
            attentions - np.sum(attentions * attentions, axis=2, keepdims=True)
    )

    # print(dc_draw)
    print(dc_draw_test)

    test_r = np.array([
        [1, 2],
        [3, 4]
    ])

    test_keys_2 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])


    # print(np.dot(test_r, test_keys_2))
    print(np.einsum('hst,htk->hsk', dc_draw, test_keys))
    print(np.einsum('hts,hsk->htk', dc_draw, test_keys))

    # print(np.tensordot(dc_dk[0], test_key_weights[0].T, axes=[1, 0]))
    # print(np.sum(np.einsum('htk,hik->hti', dc_dk, test_key_weights), axis=0))

    # print(np.tensordot(dc_dk[0], test_key_weights[0].T, axes=[1, 0]))

    # print(np.tensordot(dc_da, dc_dv, axes=[0, 1]))
    # print(np.einsum('ti,htv->hiv', dc_da, dc_dv))
    # print(np.einsum('tv,htv->hiv', dc_da, dc_dv))

    # output = np.einsum('hts,hsv->htv', attentions, test_queries)
    # print(output)
    # concat_output = output.transpose(1, 0, 2).reshape(output.shape[1], -1)
    # print(concat_output)

    # print(test[:, :, np.newaxis])
    #
    # print(np.sum((test[:, :, np.newaxis] * values), axis=1))
    # print()
    # print(np.dot(test[0], values[0]))
    # print(np.dot(dc_dr.T, dr_dq))


    # test = np.array([
    #     [1, 2, 5],
    #     [1, 3, 3],
    #     [1, 2, 9]
    # ])
    #
    # attention = np.array([
    #     [1, 2, 3],
    #     [1, 4, 3],
    #     [1, 2, 3]
    # ])
    #
    # print(np.sum(attention[:, :, np.newaxis] * test[:, np.newaxis], axis=0))
    #
    # print(np.dot(attention.T, test))


    # print(np.outer(values, dc_da, axis=1))

    # print(np.outer(np.array([3, 4]), np.array([6, 7])))
    #
    # print(values[:, :, np.newaxis] * dc_da[:, np.newaxis, :])

    # n, m = values.shape
    # idx = np.arange(m)
    # out = np.zeros((n, m, m))
    # out[:, idx, idx] = values
    #
    # print(out)

    # print(np.sum(tokens[:, :, np.newaxis] * dc_dvs[:, np.newaxis], axis=0))

    # print(np.tensordot(queries, keys, axes=[1, 1]))

    # num tokens: 3
    # query-token size: 3
    # value size: 2

    # attention_scores = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5],
    #     [5, 6, 7],
    # ])

    # dc_da = np.array([
    #     [1, 2],
    #     [3, 4],
    #     [5, 6],
    # ])
    #
    # values = np.array([
    #     [3, 4],
    #     [4, 5],
    #     [5, 6],
    # ])

    # queries = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    # ])
    #
    # keys = np.array([
    #     [2, 4, 6],
    #     [3, 5, 7],
    # ])
    #
    # dc_draw = np.array([
    #     [1, 2],
    #     [3, 4]
    # ])
    #
    # dc_dquery = np.dot(dc_draw, keys)
    # dc_dkey = np.dot(dc_draw.T, queries)
    #
    # print(dc_dquery)
    # print(dc_dkey)


    # prev_layer_a = np.array([
    #     [1, 2, 3],
    #     [3, 4, 5],
    #     [5, 6, 7],
    # ])

    # print(np.dot(prev_layer_a.T, dc_dv))
    # print(np.dot(values, dc_da.T))

    # test_input = np.array([
    #     [1, 2],
    #     [3, 4],
    #     [5, 6],
    #     [7, 8]
    # ])
    #
    # test_dc_da = np.array([
    #     [1, 2, 3],
    #     [1, 2, 4],
    #     [1, 2, 3],
    #     [1, 2, 3],
    # ])

    # print(test_dc_da[0] * test_input[0][:, np.newaxis])
    #
    # print(np.sum(test_dc_da[:, np.newaxis, :] * test_input[:, :, np.newaxis], axis=0))
    # print(np.dot(test_input.T, test_dc_da))

    # print(test_dc_da + np.array([1, 2, 3]))

    # print(test_input * test_dc_dz)

    # print(test_dc_dz * test_input.sum(axis=0)[:, np.newaxis])

    # Shape (key, query), so each row containes the values of key_i dot all queries
    # print(np.tensordot(keys, queries, axes=[1, 1]))
    # e_xs = np.exp(keys - np.max(keys, axis=1, keepdims=True))
    # softmax = e_xs / np.sum(e_xs, axis=1, keepdims=True)
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum()