import chess_board
import model
import model_functions
import layers
import random
import numpy as np


def get_ohe_board(board):
    new_board = np.zeros((8, 8, 12))
    for y in range(8):
        for x in range(8):
            piece = board[y, x]
            if piece != 0:
                new_board[y, x, piece - 1] = 1
    return new_board


def make_board_channels(board):
    board = np.transpose(board, (2, 0, 1))
    return board


def main():
    board = chess_board.Board()

    games_per_train_session = 30
    epochs_per_session = 3
    learning_rate = 0.01
    sessions = 5

    # model_1 = model.Model(
    #     model_functions.categorical_entropy,
    #     (12, 8, 8),
    #     layers.Convolution(32, (3, 3), model_functions.relu),
    #     layers.Convolution(64, (3, 3), model_functions.relu),
    #     layers.Dense(256, model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(128, model_functions.softmax)
    # )
    model_1 = model.Model.load("Models/pretrained_chess_model_1")
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
    model_2 = model.Model.load("Models/pretrained_chess_model_2")
    # model_2 = model.Model.load("Models/model_2_chess")

    # Pretrain the models on rules of the game
    # for k in range(1):
    #     positions = []
    #     moves = []
    #     for i in range(500):
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
    #             move_ohe = np.zeros(128)
    #             move_ohe[np.prod(move[0])] = 0.5
    #             move_ohe[np.prod(move[1]) + 64] = 0.5
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
    #
    #     # model_1.fit(np.array(positions), np.array(moves), 4, 0.01, 32)
    #     model_2.fit(np.array(positions), np.array(moves), 1, 0.01, 32)
    #
    #     model_1.save("Models/pretrained_chess_model_1")
    #     model_2.save("Models/pretrained_chess_model_2")
    #     print("Saved pretrain")


    for j in range(sessions):
        white_won = 0
        black_won = 0
        model_1_won = 0
        model_2_won = 0
        draw = 0

        for i in range(games_per_train_session):
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
            prev_board_4 = None

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

                temp_board_ohe = get_ohe_board(temp_board)

                if is_current_model_1:
                    move_prediction = model_1.predict(temp_board_ohe)
                else:
                    move_prediction = model_2.predict(temp_board_ohe)

                legal_moves = board.get_legal_moves(is_current_model_black)

                prev_board_4 = np.copy(prev_board_3)
                prev_board_3 = np.copy(prev_board_2)
                prev_board_2 = np.copy(prev_board)
                prev_board = np.copy(board.board)

                if random.random() < 0.05:
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
                        from_score = move_prediction[np.prod(from_move)]
                        to_score = move_prediction[np.prod(to_move) + 64]
                        total_score = from_score * to_score
                        if total_score > top_score and random.random() < 0.95   :
                            top_move = (from_move, to_move)
                            top_score = total_score

                    total_top_score += top_score
                    if is_current_model_black:
                        black_rewards.append(top_score * 4)
                    else:
                        white_rewards.append(top_score * 4)

                    if top_move is None:
                        draw += 1
                        break

                    if board.board[top_move[1]] != 0:
                        if is_current_model_black:
                            black_rewards[-1] += board.board[top_move[1]] / 6
                            if len(white_rewards) != 0:
                                white_rewards[-1] -= board.board[top_move[1]] / 6
                        else:
                            white_rewards[-1] += (board.board[top_move[1]] - 6) / 6
                            if len(black_rewards) != 0:
                                black_rewards[-1] -= (board.board[top_move[1]] - 6) / 6

                    top_move_ohe = np.zeros(128)
                    top_move_ohe[np.prod(top_move[0])] = 0.5
                    top_move_ohe[np.prod(top_move[1]) + 64] = 0.5

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
                    draw += 1
                    break

                if move_num > 300:
                    draw += 1
                    break

            # print(f"Avg top score: {total_top_score / move_num}")

            if len(black_rewards) == 0 or (len(white_rewards)) == 0:
                board.reset()
                continue

            black_rewards = np.array(black_rewards)
            white_rewards = np.array(white_rewards)

            if board.black_won:
                black_won += 1
                if is_current_model_1:
                    model_2_won += 1
                else:
                    model_1_won += 1
                black_rewards += 1.0
                # give reward for move that won game (taking king)
                if not last_move_rand:
                    black_rewards[-1] += 2.0
                white_rewards += -1.0
            elif board.game_ended:
                white_won += 1
                if is_current_model_1:
                    model_2_won += 1
                else:
                    model_1_won += 1
                black_rewards += -1.0
                white_rewards += 1.0
                # give reward for move that won game (taking king)
                if not last_move_rand:
                    white_rewards[-1] += 2.0
            else:
                black_rewards += -0.5
                white_rewards += -0.5

            board.reset()

            if is_current_model_black == is_current_model_1:
                model_1_train = black_board_positions
                model_1_labels = black_moves
                model_1_reward = black_rewards

                model_2_train = white_board_positions
                model_2_labels = white_moves
                model_2_reward = white_rewards
            else:
                model_1_train = white_board_positions
                model_1_labels = white_moves
                model_1_reward = white_rewards

                model_2_train = black_board_positions
                model_2_labels = black_moves
                model_2_reward = black_rewards

            model_1.fit(np.array(model_1_train), np.array(model_1_labels), epochs_per_session, learning_rate, 1, console_updates=False, reward_mults=model_1_reward)

            model_2.fit(np.array(model_2_train), np.array(model_2_labels), epochs_per_session, learning_rate, 1, console_updates=False, reward_mults=model_2_reward)

        print(f"Draws: {draw}")
        print(f"White wins: {white_won}")
        print(f"Black wins: {black_won}")
        print(f"Model 1 wins: {model_1_won}:")
        print(f"Model 2 wins: {model_2_won}")
        print()

    print("Saving...")
    model_1.save("Models/model_1_chess")
    model_2.save("Models/model_2_chess")


if __name__ == "__main__":
    main()