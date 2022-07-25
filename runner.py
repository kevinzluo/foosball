from rank_players import pull_table, read_info, rank_elo, rank_least_squares, print_rankings
from plotter import anim_elo_history, plot_elo_history
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ELO graphics")
    parser.add_argument("-fancy", default=0)
    parser.add_argument("-gif", default=0)
    parser.add_argument("-recency", default = 30, help = "only show players who've played in one of the last `recency` games; -1 to plot all players")
    parser.add_argument("-past_games", default = -1, help = "only plot the last `past_games` games; -1 to plot all games")

    args = parser.parse_args()
    pull_table()

    games, players, n_players, player_numbers = read_info()
    ls_scores = rank_least_squares(games, players, n_players, player_numbers)
    elo_scores, elo_history = rank_elo(players, games)
    print_rankings(ls_scores, elo_scores, players)
    for player in players:
        plot_elo_history(player, games, elo_history)
    
    if int(args.fancy) == 1:  
        anim_elo_history(players, games, elo_history, int(args.recency), int(args.past_games), int(args.gif))
