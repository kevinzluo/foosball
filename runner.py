from rank_players import pull_table, read_info, rank_elo, rank_least_squares, print_rankings
from plotter import plot_elo_history

if __name__ == "__main__":
    pull_table()
    games, players, n_players, player_numbers = read_info()
    ls_scores = rank_least_squares(games, players, n_players, player_numbers)
    elo_scores, elo_history = rank_elo(players, games)
    print_rankings(ls_scores, elo_scores, players)
    for player in players:
        plot_elo_history(player, games, elo_history)
