import numpy as np


wp = 1
wn = 2
wb = 3
wr = 4
wq = 5
wk = 6

bp = wp + 6
bn = wn + 6
bb = wb + 6
br = wr + 6
bq = wq + 6
bk = wk + 6


starting_board = np.array([[br, bn, bb, bq, bk, bb, bn, br],
                           [bp, bp, bp, bp, bp, bp, bp, bp],
                           [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                           [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                           [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                           [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                           [wp, wp, wp, wp, wp, wp, wp, wp],
                           [wr, wn, wb, wq, wk, wb, wn, wr]])


def is_black(piece):
    return piece > 6


class Board:
    def __init__(self):
        self.board = np.copy(starting_board)
        self.flip_board = self.board[::-1, ::-1]
        self.opponent_board = np.where(self.flip_board > 6, self.flip_board - 6, np.where((self.flip_board != 0) & (self.flip_board <= 6), self.flip_board + 6, self.flip_board))
        self.game_ended = False
        self.black_won = False

    def make_move(self, from_pos, to_pos):
        piece = self.board[to_pos]
        if piece == wk:
            self.game_ended = True
            self.black_won = True
        elif piece == bk:
            self.game_ended = True
        self.board[to_pos] = self.board[from_pos]
        self.board[from_pos] = 0
        piece = self.board[to_pos]
        is_piece_black = is_black(piece)
        if ((piece - 1) % 6 + 1) == wp and ((is_piece_black and to_pos[0] == 7) or (not is_piece_black and to_pos[0] == 0)):
            # Make pawn into queen if it is at the end of the board
            # Can add other options later
            self.board[to_pos] += 4

        self.flip_board = self.board[::-1, ::-1]
        self.opponent_board = np.where(self.flip_board > 6, self.flip_board - 6, np.where((self.flip_board != 0) & (self.flip_board <= 6), self.flip_board + 6, self.flip_board))

    def get_legal_moves(self, is_player_black):
        moves = []

        for y in range(self.board.shape[0]):
            for x in range(self.board.shape[1]):
                if is_player_black == is_black(self.board[y, x]):
                    from_move = (y, x)
                    to_moves = self.get_legal_moves_piece(from_move)
                    for move in to_moves:
                        moves.append((from_move, move))

        return moves

    def get_legal_moves_piece(self, piece_pos):
        moves = []
        piece = self.board[piece_pos]
        is_piece_black = is_black(piece)

        if piece != 0:
            temp_board = self.board
            if not is_piece_black:
                temp_board = self.flip_board

            to_moves = []

            t_y = piece_pos[0]
            t_x = piece_pos[1]

            true_piece = (piece - 1) % 6 + 1
            if not is_piece_black:
                t_y = 7 - t_y
                t_x = 7 - t_x

            if true_piece == wp:
                # t_y + 1 is always 7 or less. Will promote if reaches end.
                # Forward moves
                if temp_board[t_y + 1, t_x] == 0:
                    to_moves.append((t_y + 1, t_x))
                if t_y == 1 and temp_board[t_y + 1, t_x] == 0 and temp_board[t_y + 2, t_x] == 0:
                    to_moves.append((t_y + 2, t_x))

                # Diagonal taking moves
                if t_x + 1 < 8 and (
                        temp_board[t_y + 1, t_x + 1] != 0 and is_black(temp_board[t_y + 1, t_x + 1]) != is_piece_black):
                    to_moves.append((t_y + 1, t_x + 1))
                if t_x - 1 >= 0 and (
                        temp_board[t_y + 1, t_x - 1] != 0 and is_black(temp_board[t_y + 1, t_x - 1]) != is_piece_black):
                    to_moves.append((t_y + 1, t_x - 1))

            elif true_piece == wn:
                diffs = (-2, -1, 1, 2)

                for y_a in diffs:
                    for x_a in diffs:
                        if abs(y_a) != abs(x_a):
                            m_y = t_y + y_a
                            m_x = t_x + x_a
                            if 0 <= m_y < 8 and 0 <= m_x < 8:
                                take_piece = temp_board[m_y, m_x]
                                if take_piece == 0 or is_black(take_piece) != is_piece_black:
                                    to_moves.append((m_y, m_x))

            elif true_piece == wb or true_piece == wq:
                for i in reversed(range(0, t_x)):
                    check_x = i
                    check_y = t_y - (i - t_x)

                    if check_y > 7:
                        break

                    check_piece = temp_board[check_y, check_x]

                    if check_piece == 0:
                        to_moves.append((check_y, check_x))
                    elif is_black(check_piece) != is_piece_black:
                        to_moves.append((check_y, check_x))
                        break
                    else:
                        break

                for i in reversed(range(0, t_x)):
                    check_x = i
                    check_y = t_y + (i - t_x)

                    if check_y < 0:
                        break

                    check_piece = temp_board[check_y, check_x]

                    if check_piece == 0:
                        to_moves.append((check_y, check_x))
                    elif is_black(check_piece) != is_piece_black:
                        to_moves.append((check_y, check_x))
                        break
                    else:
                        break

                if t_x < 7:
                    for i in range(t_x + 1, 8):
                        check_x = i
                        check_y = t_y - (i - t_x)

                        if check_y < 0:
                            break

                        check_piece = temp_board[check_y, check_x]

                        if check_piece == 0:
                            to_moves.append((check_y, check_x))
                        elif is_black(check_piece) != is_piece_black:
                            to_moves.append((check_y, check_x))
                            break
                        else:
                            break

                    for i in range(t_x + 1, 8):
                        check_x = i
                        check_y = t_y + (i - t_x)

                        if check_y > 7:
                            break

                        check_piece = temp_board[check_y, check_x]

                        if check_piece == 0:
                            to_moves.append((check_y, check_x))
                        elif is_black(check_piece) != is_piece_black:
                            to_moves.append((check_y, check_x))
                            break
                        else:
                            break

            if true_piece == wr or true_piece == wq:
                for i in reversed(range(0, t_x)):
                    check_x = i
                    check_y = t_y

                    check_piece = temp_board[check_y, check_x]

                    if check_piece == 0:
                        to_moves.append((check_y, check_x))
                    elif is_black(check_piece) != is_piece_black:
                        to_moves.append((check_y, check_x))
                        break
                    else:
                        break

                for i in reversed(range(0, t_y)):
                    check_x = t_x
                    check_y = i

                    check_piece = temp_board[check_y, check_x]

                    if check_piece == 0:
                        to_moves.append((check_y, check_x))
                    elif is_black(check_piece) != is_piece_black:
                        to_moves.append((check_y, check_x))
                        break
                    else:
                        break

                if t_x < 7:
                    for i in range(t_x + 1, 8):
                        check_x = i
                        check_y = t_y

                        check_piece = temp_board[check_y, check_x]

                        if check_piece == 0:
                            to_moves.append((check_y, check_x))
                        elif is_black(check_piece) != is_piece_black:
                            to_moves.append((check_y, check_x))
                            break
                        else:
                            break

                if t_y < 7:
                    for i in range(t_y + 1, 8):
                        check_x = t_x
                        check_y = i

                        check_piece = temp_board[check_y, check_x]

                        if check_piece == 0:
                            to_moves.append((check_y, check_x))
                        elif is_black(check_piece) != is_piece_black:
                            to_moves.append((check_y, check_x))
                            break
                        else:
                            break

            if true_piece == wk:
                for y_a in range(-1, 2):
                    for x_a in range(-1, 2):
                        if not y_a == x_a == 0:
                            m_y = t_y + y_a
                            m_x = t_x + x_a
                            if 0 <= m_y < 8 and 0 <= m_x < 8:
                                check_piece = temp_board[m_y, m_x]
                                if check_piece == 0 or is_black(check_piece) != is_piece_black:
                                    to_moves.append((m_y, m_x))

            for move in to_moves:
                if not is_piece_black:
                    moves.append((7 - move[0], 7 - move[1]))
                else:
                    moves.append(move)

        return moves

    def reset(self):
        self.board = np.copy(starting_board)
        self.flip_board = self.board[::-1, ::-1]
        self.opponent_board = np.where(self.flip_board > 6, self.flip_board - 6, np.where((self.flip_board != 0) & (self.flip_board <= 6), self.flip_board + 6, self.flip_board))
        self.black_won = False
        self.game_ended = False

    def print_board(self):
        for y in range(8):
            print(self.board[y])



