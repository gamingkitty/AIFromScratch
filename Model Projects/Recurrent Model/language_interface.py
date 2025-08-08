from scratch_model import model
import numpy as np
import re


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
    context = 15

    data, labels, vocab = load_data("Training Data/Conversations/conversations.txt", context)
    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}
    known_tokens = word_to_index.keys()

    language_model = model.Model.load("Models/embedding_conversation")

    print(f"Model param num: {language_model.get_param_num()}")

    chat_history = [0 for _ in range(context)]
    while True:
        user_prompt = ">personb< " + input("> ") + " >endoftext<\n>persona<"
        input_tokens = tokenize(user_prompt)
        not_in_vocab = False
        for token in input_tokens:
            if token not in known_tokens:
                print("One or more tokens not in vocabulary: " + token)
                not_in_vocab = True
                break

        if not_in_vocab:
            continue

        chat_history += [word_to_index[token] for token in input_tokens]

        while len(chat_history) > context:
            chat_history.pop(0)

        ai_response = []
        num = 0
        while num < 30:
            # Currently just do max rather than probability distribution
            next_token = int(np.argmax(language_model.predict(np.array(chat_history))))

            token_string = index_to_word[next_token]
            if token_string == ">endoftext<":
                break

            ai_response.append(next_token)

            chat_history.append(next_token)
            while len(chat_history) > context:
                chat_history.pop(0)

            num += 1

        # chat_history += word_to_index['\n']
        while len(chat_history) > context:
            chat_history.pop(0)

        ai_response_string = ""
        for token in ai_response:
            ai_response_string += index_to_word[token] + " "
        print(ai_response_string + "\n")


if __name__ == "__main__":
    main()
