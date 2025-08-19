from scratch_model import *
import re
import numpy as np


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

    # for i in range(context, len(tokens), context):
    #     prev_words = []
    #     for j in range(context):
    #         prev_words.insert(0, token_to_index[tokens[i - j - 1]])
    #     data.append(prev_words)
    #     label_arr = prev_words[1:] + [token_to_index[tokens[i]]]
    #     labels.append(np.array([np.eye(vocab_size)[token] for token in label_arr]))

    return data, labels, vocab


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def main():
    embedding_dimension = 128
    learning_rate = 0.008
    epochs = 5

    print("Loading Data...")
    data, labels, vocab = load_data("Training Data/Conversations/conversations.txt")
    print("Data loaded!")

    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}

    print(f"Data length: {len(data)}")
    print(f"Vocab size: {vocab_size}")

    # ai_model = Model(
    #     model_functions.cross_entropy,
    #     (-1,),
    #     [
    #         layers.Embedding(embedding_dimension, vocab_size, model_functions.linear),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Stack(10),
    #         layers.Loop(
    #             layers.Dense(512, model_functions.relu),
    #             layers.Dense(256, model_functions.relu),
    #             layers.Dense(vocab_size, model_functions.softmax)
    #         )
    #     ],
    #     accuracy_function=accuracy
    # )
    # print(f"Param num: {ai_model.get_param_num()}")

    ai_model = Model.load("Models/conversation_recurrent")

    initial_accuracy = ai_model.test(data, labels)
    print(f"Initial accuracy: {initial_accuracy * 100:.4}%")

    ai_model.fit(data, labels, epochs, learning_rate)

    final_accuracy = ai_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.4}%")

    ai_model.save("Models/conversation_recurrent")


if __name__ == "__main__":
    main()
