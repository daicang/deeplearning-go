import random
import math

from base import Agent
from dlgo.gotypes import Player
from dlgo.goboard import GameState


class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children: list[MCTSNode] = []
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        i = random.randint(0, len(self.unvisited_moves) - 1)
        move = self.unvisited_moves.pop(i)
        next_state = self.game_state.apply_move(move)
        child_node = MCTSNode(next_state, self, move)
        self.children.append(child_node)
        return child_node

    def record_win(self, winner):
        self.num_rollouts += 1
        self.win_counts[winner] += 1

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminate(self):
        return self.game_state.is_over()

    def winning_fraction(self, player):
        if self.num_rollouts == 0:
            return 0
        return float(self.win_counts[player]) / self.num_rollouts


def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exploration


class MCTSAgent(Agent):

    def select_child(self, node: MCTSNode):
        total_rollouts = sum([child.num_rollouts for child in node.children])

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_fraction(node.game_state.next_player),
                self.temperature,
            )
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def select_move(self, game_state: GameState):
        root = MCTSNode(game_state)
        for i in range(self.num_simulations):
            node = root
            while (not node.can_add_child()) and (not node.is_terminate()):
                node = self.select_child()

            if node.can_add_child():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            percent = child.winning_fraction(game_state.next_player)
            if percent > best_pct:
                best_pct = percent
                best_move = child.move
        return best_move
