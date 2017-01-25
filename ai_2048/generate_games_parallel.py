"""
Generate random 2048 games and save them.
You can set which games will be saved by their final score or highest tile value.
Games are played parallel.
"""

from board_gym import GymBoard
import numpy as np
from util import Step, Game
import pickle
from joblib import Parallel, delayed
from math import ceil
import os
from util import encode_action
import random
import string


def gen_games(n_games, job_id, n_jobs, weights=None, min_score=0, min_tile_value=0, verbose=0):
    games = []
    i = 0

    if not weights:
        weights = [0.25, 0.25, 0.25, 0.25]

    play_for_score = False
    play_for_tile = False
    play_for_both = False

    if min_score and not min_tile_value:
        play_for_score = True
    if min_tile_value and not min_score:
        play_for_tile = True
    if min_score and min_tile_value:
        play_for_both = True
    if not min_score and not min_tile_value:
        raise ValueError("min_score or min_tile_value or both must be > 0!")

    try:
        while len(games) != n_games:
            env = GymBoard()
            r = np.random.RandomState()
            steps = []

            while True:
                matrix = env.matrix
                action = r.choice(actions, p=weights)
                moved = env.move(action)

                if moved:
                    steps.append(Step(matrix=matrix, action=action, action_encoded=encode_action(action)))

                if env.is_gameover():
                    if (play_for_score and env.score >= min_score) or \
                       (play_for_tile and np.any(env.matrix >= min_tile_value)) or \
                       (play_for_both and env.score >= min_score and np.any(env.matrix >= min_tile_value)):
                        if verbose:
                            print("\tchunk {}/{}...game {}/{}".format(job_id, n_jobs, i + 1, n_games))
                        games.append(Game(steps=steps, score=env.score, random_seed=env.random_seed, is_gameover=True))
                        i += 1
                    break
    except KeyboardInterrupt:
        pass

    return games


### OPTIONS ####################################################################
#root_dir = "/storage/brno2/home/jirinovo/_school/mvi/2048"
root_dir = "/home/jirka/ownCloud-symlinks/_05_patak_zimni/MVI/_semestralka/2048_NN/game_2048"
n_games = int(1e5)
n_jobs = 4
actions = [0, 1, 2, 3]
min_score = 0
min_tile_value = 1024
# after playing n games, save them to one file
n_save_after = 10
verbose = 1
# random string to add to each games chunk file
random_string_size = 3

# Probabilities to random pick actions. When False, actions are taken from uniform distribution, e.g. [0.25, 0.25, 0.25, 0.25].
# Actions: "up": 0, "down": 1, "left": 2, "right": 3
weights = None
# this should try to keep the highest value tiles in left upper corner
#weights = [0.4, 0.05, 0.4, 0.15]

base_dir = "random_games-{chunk_size}_games_per_chunk-{min_score}_min_score-{min_tile_value}_min_tile_value-weights_{weights}".format(
    chunk_size=n_save_after, min_score=min_score, min_tile_value=min_tile_value, weights=weights)
games_file_template = "{root_dir}/data/{base_dir}/random_games-chunk_{{chunk}}-{{random_string}}".format(root_dir=root_dir, base_dir=base_dir)
################################################################################

games = []
n_per_job = ceil(n_save_after / n_jobs)
modulo = n_save_after % n_jobs
n_chunks = ceil(n_games / n_save_after)
n_per_chunk = n_games // n_chunks

if modulo == 0:
    n_list = [n_per_job] * n_jobs
else:
    n_per_jobs = [n_per_job] * (n_jobs - 1)
    n_list = n_per_jobs + [n_save_after - sum(n_per_jobs)]

n_list = list(zip(n_list, [i+1 for i in range(len(n_list))]))
flatten = lambda l: [item for sublist in l for item in sublist]
get_random_string = lambda n: ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(n))

try:
    os.mkdir("{root_dir}/data/{base_dir}".format(root_dir=root_dir, base_dir=base_dir))
except FileExistsError:
    pass

for i in range(n_chunks):
    print("chunk {}/{}".format(i+1, n_chunks))

    games = Parallel(n_jobs=n_jobs)(
        delayed(gen_games)(n[0],
                           n[1],
                           n_jobs,
                           weights=weights,
                           min_score=min_score,
                           min_tile_value=min_tile_value,
                           verbose=verbose) for n in n_list)
    games = flatten(games)
    games_file = games_file_template.format(chunk=i + 1 if i > 9 else "0" + str(i),
                                            random_string=get_random_string(random_string_size))
    with open(games_file, mode="wb") as f:
        pickle.dump(games, f)
