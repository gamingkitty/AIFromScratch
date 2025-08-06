import time
import textwrap
import numpy as np
import re
from scratch_model import model
import sys


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
    word_data, labels, vocab = load_data("embedding_data_test.txt", 11)
    vocab_size = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    embedding_model = model.Model.load("Models/prediction_model_3")

    while True:
        sentence = input("Get model prediction for: ")
        words = sentence.split(" ")
        words = [word.lower() for word in words]
        if len(words) > 10:
            print("Too many words!")
            continue

        do_reset = False
        for word in words:
            if word not in vocab:
                print("One or more words not in vocabulary!")
                do_reset = True
                break
        if do_reset:
            continue

        model_input = np.array([])
        # for i in range(len(words) // 2):
        #     current_input = np.zeros(len(word_to_index))
        #     current_input[word_to_index[words[i]]] = 1
        #     model_input.append(current_input)
        #
        # for i in range(len(words) // 2 + 1, len(words)):
        #     current_input = np.zeros(len(word_to_index))
        #     current_input[word_to_index[words[i]]] = 1
        #     model_input.append(current_input)

        for i in range(10):
            current_input = np.zeros(len(word_to_index))
            if len(words) + i >= 10:
                current_input[word_to_index[words[i + (len(words) - 10)]]] = 1
            model_input = np.append(model_input, current_input)

        model_input = model_input.reshape((10, vocab_size))

        while True:
            model_prediction = embedding_model.predict(model_input)
            predicted_word = vocab[np.random.choice(len(model_prediction), p=model_prediction)]

            if predicted_word in {".", "!", "?"}:
                sentence += predicted_word
                break
            elif predicted_word == "," or predicted_word == ";" or predicted_word == "'" or predicted_word == "\"":
                if predicted_word == "'":
                    sentence = sentence.rstrip(" ")
                sentence += predicted_word
            else:
                sentence += " " + predicted_word

            # new_input = model_prediction
            new_input = np.zeros(vocab_size)
            new_input[word_to_index[predicted_word]] = 1
            model_input = np.append(model_input[1:], new_input).reshape(10, vocab_size)

        print("Generated sentence: ")
        wrapped_sentence = textwrap.fill(sentence, width=80)
        print(wrapped_sentence)
        print()


if __name__ == "__main__":
    main()
