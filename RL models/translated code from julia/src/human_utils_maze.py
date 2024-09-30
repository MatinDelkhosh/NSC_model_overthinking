import sqlite3
import pandas as pd
import numpy as np
from src.ToPlanOrNotToPlan import *
from walls_baselines import *
from walls import *

adict = {"[\"Up\"]": 3, "[\"Down\"]": 4, "[\"Right\"]": 1, "[\"Left\"]": 2}

def find_seps(string):
    seps = [
        [m.start() for m in re.finditer(r"]],[[", string)],
        [m.start() for m in re.finditer(r"]],[]", string)],
        [m.start() for m in re.finditer(r"[],[[", string)],
        [m.start() for m in re.finditer(r"[],[]", string)]
    ]
    return sorted([s[2] for s in seps])

def get_wall_rep(wallstr, arena):
    seps = find_seps(wallstr)
    columns = [
        wallstr[2:seps[3] - 1],
        wallstr[seps[3] + 1:seps[7] - 1],
        wallstr[seps[7] + 1:seps[11] - 1],
        wallstr[seps[11] + 1:-1]
    ]
    subseps = [[0] + find_seps(col) + [len(col) + 1] for col in columns]

    wdict = {"[\"Top\"]": 3, "[\"Bottom\"]": 4, "[\"Right\"]": 1, "[\"Left\"]": 2}

    new_walls = np.zeros((16, 4))
    for i, col in enumerate(columns):
        for j in range(4):
            ind = state_ind_from_state(arena, [i + 1, j + 1])[0]
            s1, s2 = subseps[i][j:j + 2]
            entries = col[s1 + 1:s2 - 1].split(",")
            for entry in entries:
                if len(entry) > 0:
                    new_walls[ind, wdict[entry]] = 1
    return new_walls

def extract_maze_data(db_path, user_id, Larena, T=100, max_RT=5000, game_type="play", skip_init=1, skip_finit=0):
    Nstates = Larena ** 2

    conn = sqlite3.connect(db_path)
    epis = pd.read_sql_query(f"SELECT * FROM episodes WHERE user_id = {user_id} AND game_type = '{game_type}'", conn)

    if "attention_problem" in epis.columns:
        atts = epis["attention_problem"]
        keep = atts[atts == "null"].index
        epis = epis.loc[keep]

    ids = epis["id"]

    # Ensure each episode has at least 2 steps
    stepnums = [pd.read_sql_query(f"SELECT * FROM steps WHERE episode_id = {id}", conn).shape[0] for id in ids]
    ids = ids[stepnums > 1]

    inds = range(skip_init, len(ids) - skip_finit)
    ids = ids.iloc[inds]

    batch_size = len(ids)

    rews, actions, times = np.zeros((batch_size, T)), np.zeros((batch_size, T)), np.zeros((batch_size, T))
    states = np.ones((2, batch_size, T))
    trial_nums, trial_time = np.zeros((batch_size, T)), np.zeros((batch_size, T))
    wall_loc, ps = np.zeros((16, 4, batch_size)), np.zeros((16, batch_size))

    for b in range(batch_size):
        steps = pd.read_sql_query(f"SELECT * FROM steps WHERE episode_id = {ids.iloc[b]}", conn)
        trial_num = 1
        t0 = 0

        wall_loc[:, :, b] = get_wall_rep(epis.iloc[inds[b]]["walls"], Larena)
        ps[:, b] = onehot_from_state(Larena, [int(x) + 1 for x in epis.iloc[inds[b]]["reward"][1:3]])

        Tb = steps.shape[0]

        for i in range(Tb - 1, -1, -1):
            t = steps.loc[i, "step"]
            if (t > 0) and (i < Tb - 1 or steps.loc[i, "action_time"] < 20000):
                times[b, t] = steps.loc[i, "action_time"]
                rews[b, t] = int(steps.loc[i, "outcome"] == "[\"Hit_reward\"]")
                actions[b, t] = adict[steps.loc[i, "action_type"]]
                states[:, b, t] = [int(steps.loc[i, "agent"].split()[1]) + 1, int(steps.loc[i, "agent"].split()[3]) + 1]

                trial_nums[b, t] = trial_num
                trial_time[b, t] = t - t0
                if rews[b, t] > 0:
                    trial_num += 1
                    t0 = t

    RTs = np.hstack([times[:, :1], times[:, 1:T] - times[:, :T - 1]])
    RTs[RTs < 0] = np.nan
    for b in range(batch_size):
        rewtimes = np.where(rews[b, :T] > 0)[0]
        RTs[b, rewtimes + 1] -= 8 * 50

    shot = np.full((Nstates, states.shape[1], states.shape[2]), np.nan)
    for b in range(states.shape[1]):
        Tb = np.sum(actions[1, :] > 0)
        shot[:, b, :Tb] = onehot_from_state(Larena, states[:, b, :Tb].astype(int))

    return rews, actions, states, wall_loc, ps, times, trial_nums, trial_time, RTs, shot