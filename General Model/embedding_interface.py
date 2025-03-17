import numpy as np
import re
import model


def load_data(filename, context_window):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = re.split(r'[.!?]\s+', text.strip())

    word_list = [[re.sub(r'[^a-zA-Z0-9]', '', word).lower() for word in sentence.split()] for sentence in sentences]

    all_words = set(word for sentence in word_list for word in sentence if word)
    vocab = sorted(all_words)
    vocab_size = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    one_hot_encoded = [
        [np.eye(vocab_size)[word_to_index[word]] for word in sentence if word]
        for sentence in word_list
    ]

    data = []
    labels = []

    middle_word_index = context_window // 2

    for sentence in one_hot_encoded:
        sentence_len = len(sentence)
        index = 0
        while sentence_len - index >= context_window:
            data.append(sentence[index:index + middle_word_index] + sentence[index + middle_word_index + 1: index + 2 * middle_word_index + 1])
            labels.append(sentence[index + middle_word_index])
            index += 1

    return np.array(data), np.array(labels), vocab


def main():
    word_data, labels, vocab = load_data("embedding_data_test.txt", 5)
    vocab_size = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    embedding_model = model.Model.load("Models/embedding_model_cat_3")

    while True:
        words = input("Get model prediction for: ").split(" ")

        model_input = []
        for i in range(len(words) // 2):
            current_input = np.zeros(len(word_to_index))
            current_input[word_to_index[words[i]]] = 1
            model_input.append(current_input)

        for i in range(len(words) // 2 + 1, len(words)):
            current_input = np.zeros(len(word_to_index))
            current_input[word_to_index[words[i]]] = 1
            model_input.append(current_input)

        model_prediction = embedding_model.predict(np.array(model_input))
        print(f"Predicted word: {vocab[np.argmax(model_prediction)]}")
        print()


if __name__ == "__main__":
    main()
