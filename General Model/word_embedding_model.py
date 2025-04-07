import model
import layers
import activation_functions
import numpy as np
from numpy.linalg import norm
import re


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def most_similar(word, word_vectors, top_n=5):
    if word not in word_vectors:
        return f"Word '{word}' not in vocabulary."

    word_embedding = word_vectors[word]
    similarities = {
        other_word: cosine_similarity(word_embedding, word_vectors[other_word])
        for other_word in word_vectors if other_word != word
    }

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]


def analogy(word1, word2, word3, word_vectors, top_n=5):
    # Solves word analogy: word1 is to word2 as word3 is to ???
    if word1 not in word_vectors or word2 not in word_vectors or word3 not in word_vectors:
        return "One or more words not in vocabulary."

    analogy_vector = word_vectors[word1] - word_vectors[word2] + word_vectors[word3]

    similarities = {
        word: cosine_similarity(analogy_vector, word_vectors[word])
        for word in word_vectors if word not in {word1, word2, word3}
    }

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]


def load_data(filename, context_window):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize while keeping punctuation as separate tokens
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)

    tokens = [token.lower() for token in tokens]

    # Get unique vocabulary
    all_words = set(tokens)
    vocab = sorted(all_words)
    vocab_size = len(vocab)

    # Mapping words to indices
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Convert tokens to one-hot encoded representation
    one_hot_encoded = [np.eye(vocab_size)[word_to_index[word]] for word in tokens]

    data = []
    labels = []

    # middle_word_index = context_window // 2
    middle_word_index = context_window - 1

    word_len = len(one_hot_encoded)
    index = 0
    while word_len - index >= context_window:
        # data.append(sentence[index:index + middle_word_index] + sentence[index + middle_word_index + 1: index + 2 * middle_word_index + 1])
        data.append(one_hot_encoded[index:index + middle_word_index])
        labels.append(one_hot_encoded[index + middle_word_index])
        index += 1

    return np.array(data), np.array(labels), vocab


def main():
    context_window = 11
    embedding_dimension = 64

    word_data, labels, vocab = load_data("embedding_data_test.txt", context_window)
    vocab_size = len(vocab)

    print(f"Amount of data: {len(labels)}")

    print(f"Vocab size: {vocab_size}")
    print()

    embedding_model = model.Model.load("Models/prediction_model_3")
    # embedding_model = model.Model(
    #     (context_window - 1, vocab_size),
    #     layers.Embedding(embedding_dimension, activation_functions.relu, activation_functions.relu_derivative),
    #     layers.Dense(vocab_size, activation_functions.softmax, activation_functions.softmax_derivative)
    # )

    # embedding_weights = embedding_model.layers[0].weights
    #
    # word_to_index = {word: i for i, word in enumerate(vocab)}
    #
    # word_vectors = {word: embedding_weights[word_to_index[word]] for word in word_to_index}
    #
    # print(most_similar("pets", word_vectors, 10))

    embedding_model.fit(word_data, labels, 50, vocab_size * 0.12)

    embedding_model.save("Models/prediction_model_3")


if __name__ == "__main__":
    main()
