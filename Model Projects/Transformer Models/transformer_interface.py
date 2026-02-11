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

    token_to_index = {token: i + 1 for i, token in enumerate(vocab)}
    token_to_index[">null<"] = 0

    data = []
    labels = []

    prev_15_words = [0 for _ in range(15)]
    for token in tokens:
        if token == ">newconversation<":
            prev_15_words = [0 for _ in range(15)]
        else:
            token_index = token_to_index[token]
            if len(prev_15_words) == 15:
                data.append(np.array(prev_15_words))

                label = np.zeros(vocab_size)
                label[token_index - 1] = 1
                labels.append(label)

            prev_15_words.append(token_index)

            if len(prev_15_words) > 15:
                prev_15_words.pop(0)

    return data, labels, vocab, token_to_index


def main():
    data, labels, vocab, token_to_index = load_data("Training Data/conversations.txt")

    vocab_size = len(vocab)

    index_to_token = {i: token for i, token in enumerate(vocab)}

    print(token_to_index)

    token_num = 15

    language_model = Model.load("Models/test_causal")

    previous_tokens = [0 for _ in range(token_num)]

    while True:
        user_input = ">persona< " + input("> ") + " >endoftext<\n>personb<"
        tokens = tokenize(user_input)

        print(tokens)

        for token in tokens:
            previous_tokens.append(token_to_index[token])
            previous_tokens.pop(0)

        print([index_to_token[token - 1] for token in previous_tokens if token != 0])

        language_model_token = np.argmax(language_model.predict(np.array(previous_tokens))) + 1
        previous_tokens.append(language_model_token)
        previous_tokens.pop(0)

        language_model_text = ""

        while language_model_token != token_to_index[">endoftext<"]:
            language_model_text += index_to_token[language_model_token - 1] + " "
            language_model_token = np.argmax(language_model.predict(np.array(previous_tokens))) + 1
            previous_tokens.append(language_model_token)
            previous_tokens.pop(0)

        previous_tokens.append(token_to_index["\n"])
        previous_tokens.pop(0)

        print(language_model_text.rstrip(" "))


if __name__ == "__main__":
    main()
