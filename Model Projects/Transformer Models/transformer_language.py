from scratch_model import *
import numpy as np
import re


def tokenize(text):
    tokens = re.findall(r'>[^><]*<|\n|[\w]+|[^\w\s]', text)
    return [token.lower() for token in tokens]


# def load_data(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         text = file.read()
#
#     tokens = tokenize(text)
#
#     all_tokens = set(tokens)
#     all_tokens.remove(">newconversation<")
#     vocab = sorted(all_tokens)
#     vocab_size = len(vocab)
#
#     token_to_index = {token: i + 1 for i, token in enumerate(vocab)}
#     token_to_index[">null<"] = 0
#
#     data = []
#     labels = []
#
#     prev_15_words = [0 for _ in range(15)]
#     for token in tokens:
#         if token == ">newconversation<":
#             prev_15_words = [0 for _ in range(15)]
#         else:
#             token_index = token_to_index[token]
#             if len(prev_15_words) == 15:
#                 data.append(np.array(prev_15_words))
#
#                 label = np.zeros(vocab_size)
#                 label[token_index - 1] = 1
#                 labels.append(label)
#
#             prev_15_words.append(token_index)
#
#             if len(prev_15_words) > 15:
#                 prev_15_words.pop(0)
#
#     return data, labels, vocab


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


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def main():
    epochs = 10
    learning_rate = 0.005

    data, labels, vocab = load_data("Training Data/conversations.txt")

    embedding_dimension = 256
    attention_dimension = 512
    feed_forward_dimension = 2048

    dropout_percent = 0.1

    vocab_size = len(vocab)

    print(f"Vocab Size: {vocab_size}")
    print(f"Data Size: {len(data)}")

    language_model = Model(
        model_functions.softmax_cross_entropy,
        (-1,),
        [
            layers.Embedding(embedding_dimension, vocab_size, model_functions.linear),

            layers.Attention(attention_dimension, attention_dimension, mask=model_functions.causal_mask),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),
            layers.TimeDistributedDense(feed_forward_dimension, model_functions.relu),
            layers.TimeDistributedDense(attention_dimension, model_functions.linear),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),

            layers.Attention(attention_dimension, attention_dimension, mask=model_functions.causal_mask),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),
            layers.TimeDistributedDense(feed_forward_dimension, model_functions.relu),
            layers.TimeDistributedDense(attention_dimension, model_functions.linear),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),

            layers.Attention(attention_dimension, attention_dimension, mask=model_functions.causal_mask),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),
            layers.TimeDistributedDense(feed_forward_dimension, model_functions.relu),
            layers.TimeDistributedDense(attention_dimension, model_functions.linear),
            layers.Loop(
                layers.Dropout(dropout_percent),
                layers.LayerNorm(),
            ),

            layers.TimeDistributedDense(attention_dimension, model_functions.relu),

            layers.TimeDistributedDense(vocab_size, model_functions.vectorized_cross_softmax), 6
            #layers.Loop(
            #    layers.Dense(vocab_size, model_functions.softmax)
            #)
        ],
        accuracy_function=accuracy,
    )
    # language_model = Model.load("Models/test_distributed")

    print(f"Param num: {language_model.get_param_num()}")

    # initial_accuracy = language_model.test(data, labels)
    # print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")

    language_model.fit(data, labels, epochs, learning_rate)

    final_accuracy = language_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.2f}%")

    language_model.save("Models/test_distributed")


if __name__ == "__main__":
    main()
