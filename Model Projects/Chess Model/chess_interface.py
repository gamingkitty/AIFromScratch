import random
import sys

import pygame
from scratch_model import model
import chess_board
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
    pygame.init()
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.QUIT, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

    chess_model = model.Model.load("Models/model_2_chess_conv_n_2")

    tile_size = 80

    light_square_color = (200, 200, 200)
    dark_square_color = (30, 80, 30)

    screen = pygame.display.set_mode((8 * tile_size, 8 * tile_size))
    pygame.display.set_caption('Chess Interface')

    board = chess_board.Board()
    is_black_turn = False
    legal_moves = board.get_legal_moves(is_black_turn)

    selected_square = None
    selected_piece_legal_moves = None

    piece_sprite_sheet = pygame.image.load("Chess_Pieces_Sprites.png")
    piece_sprites = []

    possible_move_tile = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    pygame.draw.circle(possible_move_tile, (100, 100, 100, 128), (tile_size / 2, tile_size / 2), tile_size * 0.4)

    for i in range(12):
        clip_rect = pygame.Rect((0, i * 200), (200, 200))
        piece_sprites.append(pygame.transform.scale(piece_sprite_sheet.subsurface(clip_rect), (tile_size, tile_size)))
    last_moves = []
    last_positions = []
    num_games_train_on = 5
    moves = []
    positions = []
    rewards = []
    last_bot_move = None
    last_bot_position = None
    while True:
        mouse_pos = pygame.mouse.get_pos()

        is_square_light = True
        for y in range(8):
            for x in range(8):
                color = dark_square_color
                if is_square_light:
                    color = light_square_color
                if is_black_turn:
                    pygame.draw.rect(screen, color, ((7 - x) * tile_size, (7 - y) * tile_size, tile_size, tile_size))
                else:
                    pygame.draw.rect(screen, color, (x * tile_size, y * tile_size, tile_size, tile_size))
                is_square_light = not is_square_light

                # Draw pieces there
                piece = board.board[y, x]
                if piece != 0:
                    if is_black_turn:
                        screen.blit(piece_sprites[piece - 1], ((7 - x) * tile_size, (7 - y) * tile_size))
                    else:
                        screen.blit(piece_sprites[piece - 1], (x * tile_size, y * tile_size))

                if selected_square is not None and (board.board[selected_square] > 6) == is_black_turn and (y, x) in selected_piece_legal_moves:
                    if is_black_turn:
                        screen.blit(possible_move_tile, ((7 - x) * tile_size, (7 - y) * tile_size))
                    else:
                        screen.blit(possible_move_tile, (x * tile_size, y * tile_size))

            is_square_light = not is_square_light

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Saved")
                # chess_model.save("Models/")
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                square_pos = (mouse_pos[1] // tile_size, mouse_pos[0] // tile_size)
                if is_black_turn:
                    square_pos = (7 - square_pos[0], 7 - square_pos[1])
                piece = board.board[square_pos]
                if selected_square is not None:
                    if (selected_square, square_pos) in legal_moves:
                        if is_black_turn:
                            positions.append(make_board_channels(get_ohe_board(board.board)))
                        else:
                            positions.append(make_board_channels(get_ohe_board(board.opponent_board)))

                        move_ohe = np.zeros(4096)
                        if is_black_turn:
                            from_num = (selected_square[0]) * 8 + (selected_square[1])
                            to_num = (square_pos[0]) * 8 + (square_pos[1])
                        else:
                            from_num = (7 - selected_square[0]) * 8 + (7 - selected_square[1])
                            to_num = (7 - square_pos[0]) * 8 + (7 - square_pos[1])
                        move_ohe[from_num * 64 + to_num] = 1
                        moves.append(move_ohe)

                        board.make_move(selected_square, square_pos)
                        is_black_turn = not is_black_turn
                        legal_moves = board.get_legal_moves(is_black_turn)

                        if is_black_turn:
                            temp_board_ohe = get_ohe_board(board.board)
                        else:
                            temp_board_ohe = get_ohe_board(board.opponent_board)

                        move_prediction = chess_model.predict(make_board_channels(temp_board_ohe))

                        top_move = None
                        top_score = -1
                        total_legal_score = 0
                        for move in legal_moves:
                            if is_black_turn:
                                from_num = (move[0][0]) * 8 + (move[0][1])
                                to_num = (move[1][0]) * 8 + (move[1][1])
                            else:
                                from_num = (7 - move[0][0]) * 8 + (7 - move[0][1])
                                to_num = (7 - move[1][0]) * 8 + (7 - move[1][1])
                            total_score = move_prediction[from_num * 64 + to_num]
                            if total_score > top_score:
                                top_move = move
                                top_score = total_score
                            total_legal_score += total_score

                        print(f"Legal score: {total_legal_score}")
                        print(f"Illegal score: {1 - total_legal_score}")

                        print(f"Move top score: {top_score}")

                        board.make_move(top_move[0], top_move[1])
                        is_black_turn = not is_black_turn
                        legal_moves = board.get_legal_moves(is_black_turn)

                if piece != 0:
                    selected_square = square_pos
                    selected_piece_legal_moves = board.get_legal_moves_piece(square_pos)
                else:
                    selected_square = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    last_positions.append(positions)
                    last_moves.append(moves)
                    if len(last_positions) > num_games_train_on:
                        last_positions.pop(0)
                        last_moves.pop(0)
                    train_positions = []
                    train_moves = []
                    for positions_i, moves_i in zip(last_positions, last_moves):
                        train_positions += positions_i
                        train_moves += moves_i
                    print(train_positions[0])
                    print()
                    print(train_positions[1])
                    chess_model.fit(np.array(train_positions), np.array(train_moves), 1, 0.01)
                    positions = []
                    moves = []
                    board.reset()
                    selected_square = None
                    selected_piece_legal_moves = None
                    is_black_turn = False
                    legal_moves = board.get_legal_moves(is_black_turn)
                    if random.random() < 0.5:
                        print("You are black this time")
                        if is_black_turn:
                            temp_board_ohe = get_ohe_board(board.board)
                        else:
                            temp_board_ohe = get_ohe_board(board.opponent_board)

                        move_prediction = chess_model.predict(make_board_channels(temp_board_ohe))

                        top_move = None
                        top_score = -1
                        total_legal_score = 0
                        for move in legal_moves:
                            if is_black_turn:
                                from_num = (move[0][0]) * 8 + (move[0][1])
                                to_num = (move[1][0]) * 8 + (move[1][1])
                            else:
                                from_num = (7 - move[0][0]) * 8 + (7 - move[0][1])
                                to_num = (7 - move[1][0]) * 8 + (7 - move[1][1])
                            total_score = move_prediction[from_num * 64 + to_num]
                            if total_score > top_score:
                                top_move = move
                                top_score = total_score
                            total_legal_score += total_score

                        print(f"Legal score: {total_legal_score}")
                        print(f"Illegal score: {1 - total_legal_score}")

                        print(f"Move top score: {top_score}")

                        board.make_move(top_move[0], top_move[1])
                        is_black_turn = not is_black_turn
                        legal_moves = board.get_legal_moves(is_black_turn)
                    else:
                        print("You are white this time")


        pygame.display.flip()


if __name__ == "__main__":
    main()
