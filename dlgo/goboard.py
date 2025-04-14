import copy
import enum

from dlgo import zobrist
from dlgo.gotypes import Player, Point


class GameResult(enum.Enum):
    win = 1
    lose = 2
    draw = 3


class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert point is not None or is_pass or is_resign
        self.point: Point = point
        self.is_play: bool = (self.point is not None)
        self.is_pass: bool = is_pass
        self.is_resign: bool = is_resign

    @classmethod
    def play(cls, point) -> "Move":
        return Move(point=point)

    @classmethod
    def pass_turn(cls) -> "Move":
        return Move(is_pass=True)

    @classmethod
    def resign(cls) -> "Move":
        return Move(is_resign=True)


class GoString():
    def __init__(self, color: Player, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def without_liberty(self, point: Point) -> "GoString":
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point: Point) -> "GoString":
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, other: "GoString") -> "GoString":
        assert other.color == self.color
        combined_stones = self.stones | other.stones
        combined_liberties = self.liberties | other.liberties - combined_stones
        return GoString(self.color, combined_stones, combined_liberties)

    @property
    def num_liberties(self) -> int:
        return len(self.liberties)

    def __eq__(self, other) -> bool:
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties


class Board():
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD

    def zhash(self):
        return self._hash

    def is_on_grid(self, point) -> bool:
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point: Point) -> Player:
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point: Point) -> GoString:
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def place_stone(self, player: Player, point: Point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None

        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue

            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        new_string = GoString(player, [point], liberties)

        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)

        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        self._hash ^= zobrist.HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            new_str = other_color_string.without_liberty(point)
            if len(new_str.liberties) == 0:
                self._remove_string(other_color_string)
            else:
                self._replace_string(new_str)

        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)

    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string and neighbor_string != string:
                    # Add liberty for neighbor string
                    self._replace_string(neighbor_string.with_liberty(point))
            # Remove this point
            self._grid[point] = None
            self._hash ^= zobrist.HASH_CODE[point, string.color]

    def _replace_string(self, newstring):
        for point in newstring.stones:
            self._grid[point] = newstring


class GameState():
    def __init__(self,
                 board: Board,
                 next_player: Player,
                 previous: "GameState",
                 move: Move
                 ):
        self.board: Board = board
        self.next_player: Player = next_player
        self.last_move: Move = move

        self.previous_state: "GameState" = previous
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states | {(previous.next_player, previous.board.zhash())}
            )

    def apply_move(self, move: Move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    @property
    def situation(self):
        return (self.next_player, self.board)

    def move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        return next_situation in self.previous_states

    def is_over(self) -> bool:
        if self.last_move is None:
            return False

        if self.last_move.is_resign:
            return True

        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False

        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0

    def is_valid_move(self, move):
        if self.is_over():
            return False

        if move.is_pass or move.is_resign:
            return True

        # Occupied
        if self.board.get(move.point):
            return False

        if self.is_move_self_capture(self.next_player, move):
            return False

        if self.move_violate_ko(self.next_player, move):
            return False

        return True
