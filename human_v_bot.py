
from six.moves import input

from dlgo.gotypes import Player
from dlgo.agent import RandomBot
from dlgo.goboard import Move, GameState
from dlgo.utils import print_board, point_from_coords, print_move


def main():
    board_size = 9
    game = GameState.new_game(board_size)
    bot = RandomBot()

    while not game.is_over():
        print(chr(27) + "[2J")  # Clear the screen
        print_board(game.board)
        if game.next_player == Player.black:
            human = input('-- ')
            point = point_from_coords(human)
            move = Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == '__main__':
    main()
