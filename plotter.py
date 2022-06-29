import matplotlib.pyplot as plt
from rank_players import get_game_teams
from util import ELO_GRAPH_DIR


def plot_elo_history(player, games, elo_history):
    def check_if_played(game):
        game_teams = get_game_teams(game)
        return (player in game_teams[0]) | (player in game_teams[1])

    player_games = [g_id for g_id, g in enumerate(games) if check_if_played(g)]
    player_elo = [1000]
    player_elo.extend([elo_history[g_id][player] for g_id in player_games])
    # start from 1, 0th will be initial elo
    updated_games = [0]
    updated_games.extend([g_id + 1 for g_id in player_games])

    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('elo')
    ax1.set_title('{} elo history'.format(player))
    ax1.set_xlabel('Game #ID')

    ax1.plot(updated_games, player_elo)
    fname = ELO_GRAPH_DIR / "{}_elo_history.png".format(player)
    plt.savefig(fname)



