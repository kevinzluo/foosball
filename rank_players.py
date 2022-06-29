import numpy as np
from numpy import linalg
import requests
import csv

def pull_table():
    key = "1ijei0ZhIdPiY_TazfB2JZAAoNi3CWowgPquQuLv3oSU"
    sheet_name = "games"
    csv_url = f"https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    result = requests.get(url=csv_url)
    open("games.csv", "wb").write(result.content)

def read_info():
    def extract_players(game):
        positions = [
            "team_1_player_1",
            "team_1_player_2",
            "team_2_player_1",
            "team_2_player_2",
        ]
        players = {game[position] for position in positions if game[position]}
        return players

    # Unpack games and players
    players_set = set()
    games = []

    with open('games.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            games.append(row)

    for game in games:
        players_set = players_set.union(extract_players(game))

    players = list(players_set)
    n_players = len(players)
    player_numbers = {players[i]: i for i in range(n_players)}

    return games, players, n_players, player_numbers

def rank_least_squares(games, players, n_players, player_numbers):
    # Make array of games
    games_matrix = np.zeros((len(games), n_players))
    results_array = np.zeros(len(games))
    for row, game in enumerate(games):
        games_matrix[row, player_numbers[game["team_1_player_1"]]] = 1
        games_matrix[row, player_numbers[game["team_2_player_1"]]] = -1
        games_matrix[row, player_numbers[game["team_1_player_2"]]] = 1
        games_matrix[row, player_numbers[game["team_2_player_2"]]] = -1
        results_array[row] = int(game["team_1_score"]) - int(game["team_2_score"])

    # Find Scores
    scores = linalg.lstsq(games_matrix,results_array, rcond=1)[0]
    player_scores = {players[i]: scores[i] for i in range(n_players)}
    players.sort(key=lambda p: player_scores[p], reverse=True)
    return player_scores


def rank_elo(players, games):
    def calc_expected_score(p1_rating, p2_rating, p3_rating, p4_rating):
        team_1 = (p1_rating + p2_rating) / 2
        team_2 = (p3_rating + p4_rating) / 2
        e_team_1 = 1 / (1 + 10 ** ((team_2 - team_1) / 500))
        return e_team_1
        
    def calc_true_score(score_1, score_2):
        return score_1 / (score_1 + score_2)

    def calc_adjusted_rating(true_score, expected_score, old_rating):
        return old_rating + 100 * (true_score - expected_score)

    elo_scores = {player: 1000 for player in players}

    for game in games:
        expected_score_1 = calc_expected_score(
            elo_scores[game["team_1_player_1"]],
            elo_scores[game["team_1_player_2"]],
            elo_scores[game["team_2_player_1"]],
            elo_scores[game["team_2_player_2"]],
        )
        true_score = calc_true_score(
            int(game["team_1_score"]), 
            int(game["team_2_score"])
        )
        team_1_win = int(game["team_1_score"]) - int(game["team_2_score"]) > 0
        elo_scores[game["team_1_player_1"]] = calc_adjusted_rating(
            true_score,
            expected_score_1,
            elo_scores[game["team_1_player_1"]],
        )
        elo_scores[game["team_1_player_2"]] = calc_adjusted_rating(
            true_score,
            expected_score_1,
            elo_scores[game["team_1_player_2"]],
        )
        elo_scores[game["team_2_player_1"]] = calc_adjusted_rating(
            1 - true_score,
            1 - expected_score_1,
            elo_scores[game["team_2_player_1"]],
        )
        elo_scores[game["team_2_player_2"]] = calc_adjusted_rating(
            1 - true_score,
            1 - expected_score_1,
            elo_scores[game["team_2_player_2"]],
        )
    return elo_scores

def print_rankings(ls_scores, elo_scores, players):
    print("Least-Squares Ranking:")
    players.sort(key=lambda p: ls_scores[p], reverse=True)
    for i, player in enumerate(players):
        print(f"{i + 1}: {player} ({ls_scores[player]:.3f})")
    
    print("\n ELO Ratings:")
    players.sort(key=lambda p: elo_scores[p], reverse=True)
    for i, player in enumerate(players):
        print(f"{i + 1}: {player} ({elo_scores[player]:.3f})")

if __name__ == "__main__":
    pull_table()
    games, players, n_players, player_numbers = read_info()
    ls_scores = rank_least_squares(games, players, n_players, player_numbers)
    elo_scores = rank_elo(players, games)
    print_rankings(ls_scores, elo_scores, players)