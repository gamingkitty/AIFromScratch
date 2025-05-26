import chess_board
import model
import model_functions
import layers
import random
import numpy as np


def get_ohe_board(board):
    new_board = np.zeros((8, 8, 7))
    for y in range(8):
        for x in range(8):
            piece = board[y, x]
            if piece != 0:
                true_piece = (piece - 1) % 6 + 1
                new_board[y, x, true_piece] = 1
                if piece > 6:
                    new_board[y, x, 0] = 1
    return new_board


def make_board_channels(board):
    board = np.transpose(board, (2, 0, 1))
    return board


def main():
    board = chess_board.Board()

    epochs_per_session = 1
    learning_rate = 0.01
    games = 1000
    prev_games_to_train = 1

    # model_1 = model.Model(
    #     model_functions.cross_entropy,
    #     (7, 8, 8),
    #     layers.Convolution(32, (3, 3), model_functions.relu),
    #     layers.Convolution(64, (3, 3), model_functions.relu),
    #     layers.Dense(256, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(4096, model_functions.softmax)
    # )
    # model_1 = model.Model(
    #     model_functions.cross_entropy,
    #     (7, 8, 8),
    #     layers.Dense(256, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(4096, model_functions.softmax)
    # )
    # model_1 = model.Model.load("Models/pretrained_chess_model_1")
    # model_1 = model.Model.load("Models/model_1_chess")

    # model_2 = model.Model(
    #     model_functions.categorical_entropy,
    #     (12, 8, 8),
    #     layers.Convolution(32, (3, 3), model_functions.relu),
    #     layers.Convolution(64, (3, 3), model_functions.relu),
    #     layers.Dense(256, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(128, model_functions.softmax)
    # )
    # model_2 = model.Model(
    #     model_functions.cross_entropy,
    #     (7, 8, 8),
    #     layers.Dense(256, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(4096, model_functions.softmax)
    # )
    # model_2 = model.Model.load("Models/pretrained_chess_model_2")
    # model_2 = model.Model.load("Models/model_2_chess")

    model_1 = model.Model.load("Models/model_1_chess_conv_n")
    model_2 = model.Model.load("Models/model_2_chess_conv_n")

    # Pretrain the models on rules of the game
    # for k in range(1):
    #     positions = []
    #     moves = []
    #     for i in range(200):
    #         is_black = False
    #         for j in range(60):
    #             legal_moves = board.get_legal_moves(False)
    #             if is_black:
    #                 board_ohe = make_board_channels(get_ohe_board(board.board))
    #             else:
    #                 board_ohe = make_board_channels(get_ohe_board(board.opponent_board))
    #
    #             if len(legal_moves) == 0:
    #                 break
    #
    #             rand_move = random.choice(legal_moves)
    #
    #             positions.append(board_ohe)
    #             if is_black:
    #                 move = rand_move
    #             else:
    #                 # Have to rotate if it is white, as models always see from one perspective of the board only.
    #                 move = ((7 - rand_move[0][0], 7 - rand_move[0][1]), (7 - rand_move[1][0], 7 - rand_move[1][1]))
    #
    #             # move_ohe = np.zeros(128)
    #             # move_ohe[move[0][0] * 8 + move[0][1]] = 0.5
    #             # move_ohe[move[1][0] * 8 + move[1][1] + 64] = 0.5
    #             # moves.append(move_ohe)
    #             move_ohe = np.zeros(4096)
    #             from_num = move[0][0] * 8 + move[0][1]
    #             to_num = move[1][0] * 8 + move[1][1]
    #             move_ohe[from_num * 64 + to_num] = 1
    #             moves.append(move_ohe)
    #
    #             board.make_move(rand_move[0], rand_move[1])
    #
    #             if board.game_ended:
    #                 break
    #
    #             is_black = not is_black
    #
    #         board.reset()
    #
    #     accuracy = model_1.test(positions, moves)
    #     print(f"Initial Accuracy: {accuracy * 100:.6f}%")
    #     print(f"Data size: {len(positions)}")
    #
    #     model_1.fit(np.array(positions), np.array(moves), 1, 0.01, 1)
    #     # model_2.fit(np.array(positions), np.array(moves), 3, 0.01, 1)
    #
    #     model_1.save("Models/pretrained_chess_model_1")
    #     # model_2.save("Models/pretrained_chess_model_2")
    #     print("Saved pretrain")

    model_1_train_total = []
    model_1_labels_total = []
    model_1_reward_total = []
    model_2_train_total = []
    model_2_labels_total = []
    model_2_reward_total = []

    for j in range(games):

        is_current_model_1 = random.getrandbits(1) == 1
        is_current_model_black = False

        white_board_positions = []
        white_moves = []

        black_board_positions = []
        black_moves = []

        move_num = 0

        prev_board = None
        prev_board_2 = None
        prev_board_3 = None

        last_move_rand = False

        black_rewards = []
        white_rewards = []
        total_top_score = 0
        while not board.game_ended:
            move_num += 1

            if is_current_model_black:
                temp_board = board.board
            else:
                temp_board = board.opponent_board

            temp_board_ohe = make_board_channels(get_ohe_board(temp_board))

            if is_current_model_1:
                move_prediction = model_1.predict(temp_board_ohe)
            else:
                move_prediction = model_2.predict(temp_board_ohe)

            legal_moves = board.get_legal_moves(is_current_model_black)

            prev_board_4 = np.copy(prev_board_3)
            prev_board_3 = np.copy(prev_board_2)
            prev_board_2 = np.copy(prev_board)
            prev_board = np.copy(board.board)

            if random.random() < 0.1:
                move = random.choice(legal_moves)
                board.make_move(move[0], move[1])
                last_move_rand = True
            else:
                last_move_rand = False
                top_move = None
                top_score = -1
                for move in legal_moves:
                    from_move = move[0]
                    to_move = move[1]
                    if not is_current_model_black:
                        from_move = (7 - from_move[0], 7 - from_move[1])
                        to_move = (7 - to_move[0], 7 - to_move[1])
                    total_score = move_prediction[(from_move[0] * 8 + from_move[1]) * 64 + to_move[0] * 8 + to_move[1]]
                    if total_score > top_score and (top_move is None or random.random() < 0.8):
                        top_move = (from_move, to_move)
                        top_score = total_score

                if is_current_model_black:
                    black_rewards.append(-top_score * 0.3)
                else:
                    white_rewards.append(-top_score * 0.3)

                if top_move is None:
                    break

                if board.board[top_move[1]] != 0:
                    if is_current_model_black:
                        black_rewards[-1] += board.board[top_move[1]] / 6
                        # if len(white_rewards) != 0:
                        #     white_rewards[-1] -= board.board[top_move[1]] / 6
                    else:
                        white_rewards[-1] += (board.board[top_move[1]] - 6) / 6
                        # if len(black_rewards) != 0:
                        #     black_rewards[-1] -= (board.board[top_move[1]] - 6) / 6

                top_move_ohe = np.zeros(4096)
                top_move_ohe[(top_move[0][0] * 8 + top_move[0][1]) * 64 + top_move[1][0] * 8 + top_move[1][1]] = 1.0

                if is_current_model_black:
                    black_board_positions.append(temp_board_ohe)
                    black_moves.append(top_move_ohe)
                else:
                    white_board_positions.append(temp_board_ohe)
                    white_moves.append(top_move_ohe)

                if is_current_model_black:
                    board.make_move(top_move[0], top_move[1])
                else:
                    board.make_move((7 - top_move[0][0], 7 - top_move[0][1]),
                                      (7 - top_move[1][0], 7 - top_move[1][1]))

                is_current_model_1 = not is_current_model_1
                is_current_model_black = not is_current_model_black

                if np.array_equal(board.board, prev_board_4):
                    break

                if move_num > 300:
                    break

        if len(black_rewards) == 0 or (len(white_rewards)) == 0:
            board.reset()
            continue

        # Testing making rewards higher to see if it gives a better result
        black_rewards = np.array(black_rewards)
        white_rewards = np.array(white_rewards)

        if board.black_won:
            print("Black won")
            if is_current_model_1 == is_current_model_black:
                print("Model 1 Won")
            else:
                print("Model 2 Won")
            black_rewards += 1.2
            # give reward for move that won game (taking king)
            if not last_move_rand:
                black_rewards[-1] += 1.0
            white_rewards += -0.5
        elif board.game_ended:
            print("White Won")
            if is_current_model_1 != is_current_model_black:
                print("Model 1 Won")
            else:
                print("Model 2 Won")
            black_rewards += -0.5
            white_rewards += 1.2
            # give reward for move that won game (taking king)
            if not last_move_rand:
                white_rewards[-1] += 1.0
        else:
            print("Draw")
            black_rewards += -0.2
            white_rewards += -0.2
            black_rewards[-1] += -1.0
            white_rewards[-1] += -1.0

        board.reset()

        if is_current_model_black == is_current_model_1:
            model_1_train = black_board_positions
            model_1_labels = black_moves
            model_1_reward = list(black_rewards)

            model_2_train = white_board_positions
            model_2_labels = white_moves
            model_2_reward = list(white_rewards)
        else:
            model_1_train = white_board_positions
            model_1_labels = white_moves
            model_1_reward = list(white_rewards)

            model_2_train = black_board_positions
            model_2_labels = black_moves
            model_2_reward = list(black_rewards)

        model_1_train_total.append(model_1_train)
        model_1_labels_total.append(model_1_labels)
        model_1_reward_total.append(model_1_reward)

        if len(model_1_train_total) > prev_games_to_train:
            model_1_train_total.pop(0)
            model_1_labels_total.pop(0)
            model_1_reward_total.pop(0)

        model_2_train_total.append(model_2_train)
        model_2_labels_total.append(model_2_labels)
        model_2_reward_total.append(model_2_reward)

        if len(model_2_train_total) > prev_games_to_train:
            model_2_train_total.pop(0)
            model_2_labels_total.pop(0)
            model_2_reward_total.pop(0)

        model_1_reward = []
        model_1_labels = []
        model_1_train = []
        for i in range(len(model_1_train_total)):
            for k in range(len(model_1_train_total[i])):
                model_1_train.append(model_1_train_total[i][k])
                model_1_labels.append(model_1_labels_total[i][k])
                model_1_reward.append(model_1_reward_total[i][k])

        model_2_reward = []
        model_2_labels = []
        model_2_train = []
        for i in range(len(model_2_train_total)):
            for k in range(len(model_2_train_total[i])):
                model_2_train.append(model_2_train_total[i][k])
                model_2_labels.append(model_2_labels_total[i][k])
                model_2_reward.append(model_2_reward_total[i][k])

        print(f"Training on {len(model_1_train)} moves")
        print()

        if len(model_1_train) > 0:
            model_1.fit(np.array(model_1_train), np.array(model_1_labels), epochs_per_session, learning_rate, 1, console_updates=False, reward_mults=np.array(model_1_reward))

        if len(model_2_train) > 0:
            model_2.fit(np.array(model_2_train), np.array(model_2_labels), epochs_per_session, learning_rate, 1, console_updates=False, reward_mults=np.array(model_2_reward))


    print("Saving...")
    model_1.save("Models/model_1_chess_conv_n_2")
    model_2.save("Models/model_2_chess_conv_n_2")


if __name__ == "__main__":
    main()