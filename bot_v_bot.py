import time

from dlgo import goboard
from dlgo import agent, gotypes
from dlgo.utils import print_board, print_move


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: agent.RandomBot(),
        gotypes.Player.white: agent.RandomBot(),
    }
    while not game.is_over():
        time.sleep(1)

        print(chr(27)+'[2J')
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
        print_board(game.board)



if __name__ == '__main__':
    main()
