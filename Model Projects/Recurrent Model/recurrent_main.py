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

    for i in range(context, len(tokens)):
        prev_words = []
        for j in range(context):
            prev_words.insert(0, token_to_index[tokens[i - j - 1]])
        data.append(prev_words)
        labels.append(np.eye(vocab_size)[token_to_index[tokens[i]]])

    return np.array(data), np.array(labels), vocab


def main():
    embedding_dimension = 20
    context = 10
    learning_rate = 0.015
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
    #     layers.Recurrent(32, model_functions.relu),
    #     layers.Dense(64, model_functions.relu),
    #     layers.Dense(vocab_size, model_functions.softmax)
    # )
    # print(f"Param num: {ai_model.get_param_num()}")

    ai_model = model.Model.load("Models/first_recurrent")

    initial_accuracy = ai_model.test(data, labels)
    print(f"Initial accuracy: {initial_accuracy * 100:.4}%")

    ai_model.fit(data, labels, epochs, learning_rate)

    final_accuracy = ai_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.4}%")

    ai_model.save("Models/first_recurrent")


if __name__ == "__main__":
    main()
