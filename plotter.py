import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
import pandas as pd
import numpy as np
import copy
from rank_players import get_game_teams
from util import ELO_GRAPH_DIR
from scipy.interpolate import make_interp_spline

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

def anim_elo_history(players, games, elo_history):
    # get elo as df with columns player names
    trajs = pd.DataFrame(elo_history).T
    # append start rating of 1000 for players of the first game
    # (hence game 0 is just everyone at 1000)
    trajs.loc[-1] = 1000
    trajs.index = trajs.index + 1
    trajs = trajs.sort_index()

    print(trajs)
    
    # this fixes indexing issues
    # the index of a game is the true number of that game
    games = copy.deepcopy(games)
    num_games = len(games)
    games.insert(0, {})

    # get first game played
    first_games = {}
    for player in players:
        for i, game in enumerate(games):
            game_teams = get_game_teams(game)
            if (player in game_teams[0]) | (player in game_teams[1]):
                first_games[player] = i 
                break

    # fixed at 30FPS
    INTERVAL = 1 / 60 * 1000 # not actually matters
    FPS = 35
    TOTAL_RUNTIME = 5 # in seconds
    FRAMES_PER_GAME = TOTAL_RUNTIME * FPS // (num_games + 1)
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize = (6, 4))

    EASE_IN_FUNC = lambda t: t ** 2
    EASE_OUT_FUNC = lambda t: (1 - t) ** 2

    BASE_LINEWIDTH = 1
    MAX_ADDL_LINEWIDTH = np.sqrt(54) - 1
    BASE_MARKERSIZE = 18
    MAX_ADDL_MARKERSIZE = 36

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # precompute splines, markers, and thicknesses
    paths = {}
    for player in players:
        SEG_RESOLUTION = 3 * FRAMES_PER_GAME
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
        
        # gen line thicknesses
        # len(xsup) - 1 total sections
        # first row used for upscaling
        # second row used for downscaling
        # collapse by maxing along axis 1 afterwards
        line_thickness = np.zeros((2, SEG_RESOLUTION * (len(xsup) - 1)))

        # sample the midpoint of every interval of [0, 1]
        delta_range, delta = np.linspace(0, 1, SEG_RESOLUTION, endpoint=False, retstep=True)
        delta_range += delta / 2

        up_section = EASE_IN_FUNC(delta_range)
        down_section = EASE_OUT_FUNC(delta_range)

        for game_num, in_game in zip(xsup, in_game_indicator):
            if game_num == 0:
                continue
            entry_index = SEG_RESOLUTION * (game_num - (first_games[player] - 1))
            if in_game:
                # queue up upscaling for this 
                line_thickness[0, entry_index - SEG_RESOLUTION:entry_index] = up_section
                # queue up downscaling for the next section
                if game_num < num_games:
                    line_thickness[1, entry_index:entry_index + SEG_RESOLUTION] = down_section

        # now generate segments
        points = np.array([xnew, spline_path]).T
        segments = np.stack((points[:-1], points[1:]), axis = 1)

        line_thickness = line_thickness.max(axis = 0)
        # line_thickness = BASE_LINEWIDTH + line_thickness * MAX_ADDL_LINEWIDTH
        # removed to use this adjustment for markers
        paths[player] = xsup, trunc_traj, marker_sizes, xnew, segments, line_thickness
    
    def return_at_time(player, time):
        xsup, trunc_traj, marker_sizes, xnew, segments, line_thickness = paths[player]
        # filter to only plot up to current time
        curr_xsup = xsup[xsup <= time]
        curr_trunc_traj = trunc_traj[xsup <= time]
        curr_marker_sizes = marker_sizes[xsup <= time]

        mask = (xnew <= time)[1:] # we look at the RH endpoint
        curr_segments = segments[mask]
        curr_line_thickness = line_thickness[mask]

        return curr_xsup, curr_trunc_traj, curr_marker_sizes, curr_segments, curr_line_thickness


    # animation for frame i
    def animate(i):
        # clear plot
        ax.cla()

        # time starts at 0 and ends at len(games) (+1 more second of freeze at end)
        curr_time_as_game = i / FRAMES_PER_GAME

        for i, player in enumerate(players):
            # filter to only plot up to current time
            curr_xsup, curr_trunc_traj, curr_marker_sizes, curr_segments, curr_line_thickness = return_at_time(player, curr_time_as_game)

            # append on point in current time to have line end with a dot
            if curr_segments.shape[0] > 0:
                ax.scatter(list(curr_xsup) + [curr_segments[-1][1][0]], list(curr_trunc_traj) + [curr_segments[-1][1][1]], label = player,
                        s = list(curr_marker_sizes) + [BASE_MARKERSIZE + curr_line_thickness[-1] * MAX_ADDL_MARKERSIZE], 
                        # alpha = 0.8,
                        color = cycle[i % 10])
            
                ax.scatter(curr_segments[:, 1, 0], curr_segments[:, 1, 1], 
                            s = [BASE_MARKERSIZE / 9 + curr_line_thickness * (MAX_ADDL_MARKERSIZE + 6/9 * BASE_MARKERSIZE)],
                            color = cycle[i % 10])
                # ax.add_collection(LineCollection(curr_segments, 
                #                 linewidths=BASE_LINEWIDTH + MAX_ADDL_LINEWIDTH * curr_line_thickness, 
                #                 antialiaseds = False,
                #                 color = cycle[i],
                #                 # alpha = 0.8,
                #                 ))
            else:
                ax.scatter([], [], label = player,
                        # alpha = 0.8,
                        color = cycle[i % 10])


        # fix x-axis 
        # possible change: sliding x/y axis
        ax.set_xlim([0, num_games])
        Y_MIN = trajs.min().min()
        Y_MAX = trajs.max().max()
        Y_RANGE = Y_MAX - Y_MIN
        ax.set_ylim([Y_MIN - Y_RANGE / 10, Y_MAX + Y_RANGE / 10])
        ax.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        fig.tight_layout()

    # animate(15)
    # plt.show()

    anim = animation.FuncAnimation(fig, animate, frames = range(1, FRAMES_PER_GAME * num_games + FPS + 1), interval = INTERVAL / 10, blit = False)
    anim.save("elo_graphs/elo.gif", animation.ImageMagickWriter(fps = FPS))
