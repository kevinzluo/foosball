from os import minor
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import pandas as pd
import numpy as np
import copy
from rank_players import get_game_teams
from util import ELO_GRAPH_DIR
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as ticker
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

    ax1.plot(updated_games, player_elo, '-o')
    fname = ELO_GRAPH_DIR / "{}_elo_history.png".format(player)
    plt.savefig(fname)
    plt.close()

def anim_elo_history(players, games, elo_history, recency, games_to_plot, gif):
    print(pd.DataFrame(elo_history))
    # get elo as df with columns player names
    trajs = pd.DataFrame(elo_history).T
    # append start rating of 1000 for players of the first game
    # (hence game 0 is just everyone at 1000)
    trajs.loc[-1] = 1000
    trajs.index = trajs.index + 1
    trajs = trajs.sort_index()
    
    # this fixes indexing issues
    # the index of a game is the true number of that game
    games = copy.deepcopy(games)
    num_games = len(games)
    games.insert(0, {})

    if recency < 0:
        recency = num_games + 1
    if games_to_plot < 0:
        games_to_plot = num_games + 1

    START_GAME = max(0, num_games - games_to_plot)

    # get first game played
    first_games = {}
    last_games = {}
    for player in players:
        for i, game in enumerate(games):
            game_teams = get_game_teams(game)
            if (player in game_teams[0]) | (player in game_teams[1]):
                if player not in first_games:
                    first_games[player] = i
                last_games[player] = i

    players = players.copy()
    players = [player for player in players if num_games - last_games[player] < recency]
    min_first_game = min([v for _, v in first_games.items()])
    START_GAME = min(min_first_game, START_GAME)
    # fixed at 30FPS
    FPS = 30
    TOTAL_RUNTIME = 5 # in seconds
    FRAMES_PER_GAME = (TOTAL_RUNTIME * FPS) // (num_games + 1)
    plt.close('all')
    # height of 6 is comfortable for 25 players, just scaling
    fig, ax = plt.subplots(1, 1, figsize = (8, max(4, 6 * len(players) / 25)))

    EASE_IN_FUNC = lambda t: t ** 2
    EASE_OUT_FUNC = lambda t: (1 - t) ** 2

    BASE_MARKERSIZE = 0
    MAX_ADDL_MARKERSIZE = 54

    ## TODO: THIS BREAKS ONCE 80 PLAYERS
    color_cycle = cm.tab20(np.linspace(0, 1, len(players), endpoint = False)) 
    linestyle_cycle = ['solid', 'dashed', 'dashdot', 'dotted']

    # precompute splines, markers, and thicknesses
    paths = {}
    for player in players:
        SEG_RESOLUTION = max(int(3 * FRAMES_PER_GAME), 40)
        # support for x axis, take game before to start at 1000
        xsup = np.arange(first_games[player] - 1, num_games + 1)
        # finer grain for spline, with one increment per frame
        xnew = np.linspace(min(xsup), max(xsup), SEG_RESOLUTION * (len(xsup) - 1) + 1)

        # get elos for the player over their support
        trunc_traj = trajs[player][xsup]
        spl = make_interp_spline(xsup, trunc_traj, min(3, len(xsup) - 1))
        spline_path = spl(xnew)

        in_game_indicator = [int(player in games[game].values()) for game in xsup]

        # gen marker sizes
        marker_sizes = np.array([BASE_MARKERSIZE + MAX_ADDL_MARKERSIZE * game for game in in_game_indicator])
        
        # for endpoint marker thickness
        # we ignore the very first point of xnew, since it always an unplayed game
        inst_thickness = np.zeros((2, SEG_RESOLUTION * (len(xsup) - 1)))

        # sample the right endpoint of every interval of [0, 1]
        delta, delta_range = np.linspace(0, 1, SEG_RESOLUTION, endpoint=False, retstep=True)
        delta_range += delta

        up_section = EASE_IN_FUNC(delta_range)
        down_section = EASE_OUT_FUNC(delta_range)

        for game_num, in_game in zip(xsup, in_game_indicator):
            if game_num == 0:
                continue
            entry_index = SEG_RESOLUTION * (game_num - (first_games[player] - 1))
            if in_game:
                # queue up upscaling for this 
                inst_thickness[0, entry_index - SEG_RESOLUTION:entry_index] = up_section

                # queue up downscaling for the next section
                if game_num < num_games:
                    inst_thickness[1, entry_index:entry_index + SEG_RESOLUTION] = down_section

        inst_thickness = inst_thickness.max(axis = 0)
        # removed to use this adjustment for markers
        paths[player] = xsup, trunc_traj, marker_sizes, xnew, spline_path, inst_thickness
    
    def return_at_time(player, time):
        xsup, trunc_traj, marker_sizes, xnew, spline_path, inst_thickness = paths[player]
        # filter to only plot up to current time
        mask = xsup <= time
        curr_xsup = xsup[mask]
        curr_trunc_traj = trunc_traj[mask]
        curr_marker_sizes = marker_sizes[mask]

        mask = xnew <= time
        curr_xnew = xnew[mask] # we look at the RH endpoint
        curr_spline_path = spline_path[mask]
        curr_inst_thickness = inst_thickness[mask[1:]]

        return curr_xsup, curr_trunc_traj, curr_marker_sizes, curr_xnew, curr_spline_path, curr_inst_thickness

    # animation for frame i
    def animate(i):
        # clear plot
        ax.cla()

        curr_time_as_game = START_GAME + i / (5 * FPS) * (num_games + 1 - START_GAME)

        for j, player in enumerate(players):
            # filter to only plot up to current time
            curr_xsup, curr_trunc_traj, curr_marker_sizes, curr_xnew, curr_spline_path, curr_inst_thickness = return_at_time(player, curr_time_as_game)

            # append on point in current time to have line end with a dot
            color = color_cycle[j % len(color_cycle)]
            linestyle = linestyle_cycle[j // len(color_cycle)]

            if curr_inst_thickness.shape[0] > 0:
                ax.scatter(list(curr_xsup) + [curr_xnew[-1]], list(curr_trunc_traj) + [curr_spline_path[-1]], 
                        s = list(curr_marker_sizes) + [BASE_MARKERSIZE + curr_inst_thickness[-1] * MAX_ADDL_MARKERSIZE], 
                        alpha = 0.8,
                        color = color)
            
                ax.plot(curr_xnew, curr_spline_path, color = color, linestyle = linestyle, label = player, alpha = 0.8)
            else:
                ax.plot([], [], color = color, linestyle = linestyle, label = player, alpha = 0.8)


        # fix x-axis 
        # possible change: sliding x/y axis
        ax.set_xlim([START_GAME, num_games])
        Y_MIN = trajs.loc[START_GAME:].min().min()
        Y_MAX = trajs.loc[START_GAME:].max().max()
        Y_RANGE = Y_MAX - Y_MIN
        ax.set_ylim([Y_MIN - Y_RANGE / 10, Y_MAX + Y_RANGE / 10])
        ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        fig.tight_layout()
    if gif == 1:
        anim = animation.FuncAnimation(fig, animate, frames = range(1, FPS * (TOTAL_RUNTIME + 3) + 1), blit = False)
        anim.save("elo_graphs/elo.gif", animation.FFMpegWriter(fps = FPS))
    else:
        animate(FPS * (TOTAL_RUNTIME + 3))
        fig.savefig("elo_graphs/elo.png", bbox_inches = "tight")