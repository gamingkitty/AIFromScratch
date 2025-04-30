import sys

import pygame
import model
import chess_board
import numpy as np

def get_ohe_board(board):
    new_board = np.zeros((8, 8, 12))
    for y in range(8):
        for x in range(8):
            piece = board[y, x]
            if piece != 0:
                new_board[y, x, piece - 1] = 1
    return new_board


def main():
    pygame.init()
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.QUIT, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

    chess_model = model.Model.load("Models/model_1_chess")

    tile_size = 80

    light_square_color = (200, 200, 200)
    dark_square_color = (30, 80, 30)

    screen = pygame.display.set_mode((8 * tile_size, 8 * tile_size))
    pygame.display.set_caption('Chess Interface')

    board = chess_board.Board()
    is_black_turn = False
    legal_moves = board.get_legal_moves(is_black_turn)

    font = pygame.font.SysFont("Calibri", 36)

    selected_square = None
    selected_piece_legal_moves = None

    piece_sprite_sheet = pygame.image.load("Chess_Pieces_Sprites.png")
    piece_sprites = []

    possible_move_tile = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    pygame.draw.circle(possible_move_tile, (100, 100, 100, 128), (tile_size / 2, tile_size / 2), tile_size * 0.4)

    for i in range(12):
        clip_rect = pygame.Rect((0, i * 200), (200, 200))
        piece_sprites.append(pygame.transform.scale(piece_sprite_sheet.subsurface(clip_rect), (tile_size, tile_size)))

    prev_board_3 = None
    prev_board_2 = None
    prev_board = None

    while True:
        mouse_pos = pygame.mouse.get_pos()

        is_square_light = True
        for y in range(8):
            for x in range(8):
                color = dark_square_color
                if is_square_light:
                    color = light_square_color
                pygame.draw.rect(screen, color, (x * tile_size, y * tile_size, tile_size, tile_size))
                is_square_light = not is_square_light

                # Draw pieces there
                piece = board.board[y, x]
                if piece != 0:
                    screen.blit(piece_sprites[piece - 1], (x * tile_size, y * tile_size))

                if selected_square is not None and (board.board[selected_square] > 6) == is_black_turn and (y, x) in selected_piece_legal_moves:
                    screen.blit(possible_move_tile, (x * tile_size, y * tile_size))

            is_square_light = not is_square_light

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                square_pos = (mouse_pos[1] // tile_size, mouse_pos[0] // tile_size)
                piece = board.board[square_pos]
                if selected_square is not None:
                    if (selected_square, square_pos) in legal_moves:
                        board.make_move(selected_square, square_pos)
                        is_black_turn = not is_black_turn
                        legal_moves = board.get_legal_moves(is_black_turn)

                        temp_board_ohe = get_ohe_board(board.board)

                        move_prediction = chess_model.predict(temp_board_ohe)

                        top_move = None
                        top_score = -1
                        total_legal_score = 0
                        for move in legal_moves:
                            from_move = move[0]
                            to_move = move[1]

                            from_score = move_prediction[np.prod(from_move)]
                            to_score = move_prediction[np.prod(to_move) + 64]
                            total_score = from_score * to_score
                            total_legal_score += total_score
                            if total_score > top_score:
                                top_move = (from_move, to_move)
                                top_score = total_score

                        total_score = 0
                        for y in range(8):
                            for x in range(8):
                                for y2 in range(8):
                                    for x2 in range(8):
                                        from_move = (y, x)
                                        to_move = (y2, x2)
                                        score = move_prediction[np.prod(from_move)] * move_prediction[np.prod(to_move) + 64]
                                        total_score += score

                        print(f"Legal score: {total_legal_score}")
                        print(f"Illegal score: {total_score - total_legal_score}")

                        print(f"Move top score: {top_score}")

                        board.make_move(top_move[0], top_move[1])
                        is_black_turn = not is_black_turn
                        legal_moves = board.get_legal_moves(is_black_turn)

                if piece != 0:
                    selected_square = square_pos
                    selected_piece_legal_moves = board.get_legal_moves_piece(square_pos)
                else:
                    selected_square = None


        pygame.display.flip()


if __name__ == "__main__":
    main()
