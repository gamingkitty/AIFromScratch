from scratch_model import model
from scratch_model import layers
from scratch_model import model_functions
import re
import numpy as np


def load_data(filename, context):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize while keeping punctuation as separate tokens
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)

    tokens = [token.lower() for token in tokens]

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
    embedding_dimension = 50
    context = 5
    learning_rate = 0.01
    epochs = 10

    data, labels, vocab = load_data("Training Data/test_data.txt", context)
    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}

    ai_model = model.Model(
        model_functions.cross_entropy,
        (context,),
        layers.Embedding(embedding_dimension, vocab_size, model_functions.linear),
        # layers.Recurrent(128, model_functions.relu),
        layers.Dense(256, model_functions.relu),
        layers.Dense(vocab_size, model_functions.softmax)
    )

    initial_accuracy = ai_model.test(data, labels)
    print(f"Initial accuracy: {initial_accuracy * 100:.4}%")

    ai_model.fit(data, labels, epochs, learning_rate)

    # ai_model.save("Models/first_recurrent")

    final_accuracy = ai_model.test(data, labels)
    print(f"Final accuracy: {final_accuracy * 100:.4}%")


if __name__ == "__main__":
    main()
