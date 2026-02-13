from scratch_model import *
import numpy as np
import re


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

    return data, labels, vocab


def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)


def main():
    data, labels, vocab = load_data("Training Data/conversations.txt")

    vocab_size = len(vocab)

    index_to_token = {i: token for i, token in enumerate(vocab)}
    token_to_index = {token: i for i, token in enumerate(vocab)}

    language_model = Model.load("Models/test_distributed")

    index = 2

    print([index_to_token[d] for d in data[1]])
    prediction = language_model.predict(data[index])
    predicted_tokens = [index_to_token[np.argmax(pred)] for pred in prediction]
    label_tokens = [index_to_token[np.argmax(l)] for l in labels[index]]

    print()
    print(predicted_tokens)
    print(label_tokens)

    model_accuracy = sum(predicted_tokens[i] == label_tokens[i] for i in range(len(label_tokens))) / len(label_tokens)
    print(f"Accuracy: {model_accuracy}")

    previous_tokens = []

    # while True:
    #     user_input = ">persona< " + input("> ") + " >endoftext<\n>personb<"
    #     tokens = tokenize(user_input)
    #
    #     for token in tokens:
    #         previous_tokens.append(token_to_index[token])
    #
    #     language_model_token = -1
    #
    #     language_model_text = ""
    #
    #     while language_model_token != token_to_index[">endoftext<"]:
    #         prediction = language_model.predict(np.array(previous_tokens))
    #         print([index_to_token[np.argmax(pred)] for pred in prediction])
    #         language_model_token = np.argmax(prediction[-1])
    #         language_model_text += index_to_token[language_model_token] + " "
    #         previous_tokens.append(language_model_token)
    #
    #     previous_tokens.append(token_to_index["\n"])
    #
    #     print(language_model_text.rstrip(" "))


if __name__ == "__main__":
    main()
