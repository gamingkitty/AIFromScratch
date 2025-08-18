from scratch_model import model
from scratch_model import layers
from scratch_model import model_functions
import re
import numpy as np


def tokenize(text):
    tokens = re.findall(r'>[^><]*<|\n|[\w]+|[^\w\s]', text)
    return [token.lower() for token in tokens]


def load_data(filename, context):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize while keeping punctuation as separate tokens
    tokens = tokenize(text)

    # Get unique vocabulary
    all_tokens = set(tokens)
    vocab = sorted(all_tokens)
    vocab_size = len(vocab)

    # Mapping words to indices
    token_to_index = {token: i for i, token in enumerate(vocab)}

    # mapped_words = [word_to_index[word] for word in tokens]

    data = []
    labels = []

    for i in range(context, len(tokens), context):
        prev_words = []
        for j in range(context):
            prev_words.insert(0, token_to_index[tokens[i - j - 1]])
        data.append(prev_words)
        label_arr = prev_words[1:] + [token_to_index[tokens[i]]]
        labels.append(np.array([np.eye(vocab_size)[token] for token in label_arr]))

    return np.array(data), np.array(labels), vocab


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def main():
    embedding_dimension = 50
    context = 15
    learning_rate = 0.01
    epochs = 5

    data, labels, vocab = load_data("Training Data/Conversations/conversations.txt", context)
    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}

    print(f"Data length: {len(data)}")
    print(f"Vocab size: {vocab_size}")

    # ai_model = model.Model(
    #     model_functions.cross_entropy,
    #     (context,),
    #     layers.Embedding(embedding_dimension, vocab_size, model_functions.linear),
    #     layers.Recurrent(64, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(vocab_size, model_functions.softmax)
    # )
    # ai_model = model.Model(
    #     model_functions.cross_entropy,
    #     (context,),
    #     [
    #         layers.Embedding(embedding_dimension, vocab_size, model_functions.linear),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Recurrent(128, model_functions.relu),
    #         layers.Loop(
    #             layers.Dense(256, model_functions.relu),
    #             layers.Dense(vocab_size, model_functions.softmax)
    #         )
    #     ],
    #     accuracy_function=accuracy
    # )
    # print(f"Param num: {ai_model.get_param_num()}")

    ai_model = model.Model.load("Models/large_recurrent")

    initial_accuracy = ai_model.test(data, labels)
    print(f"Initial accuracy: {initial_accuracy * 100:.4}%")

    ai_model.fit(data, labels, epochs, learning_rate)

    final_accuracy = ai_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.4}%")

    ai_model.save("Models/large_recurrent")


if __name__ == "__main__":
    main()
