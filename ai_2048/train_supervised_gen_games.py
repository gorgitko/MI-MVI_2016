"""
Train the NN in supervised way from game records consisting of (game state, action) pairs.
You can have running generate_games_parallel.py in background and wait for newly generated games.
After 'n' games, the NN is evaluated and compared with random play. Results are saved in pandas DataFrame.
"""

from board_gym import GymBoard
from util import save_history, train_test_stratified

from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from pprint import pprint
import os
import pickle
import pandas as pd
from time import sleep
from glob import glob
import atexit
import logging
from collections import OrderedDict


def evaluate(model, n_games, log_steps=False, random_seed=0, verbose=0, use_eps_greedy_policy=False, eps=0.1):
    """
    Parameters
    ----------
    model
    n_games
    log_steps
    random_seed
    verbose
    use_eps_greedy_policy : bool
        Whether to use epsilon greedy policy for NN prediction of actions.

    Returns
    -------
    pandas.DataFrame
    """
    games_nn = []
    games_random = []
    actions = list(range(GymBoard.NB_ACTIONS))

    for i in range(n_games):
        if verbose:
            print("game {}/{}".format(i + 1, n_games))

        # NN play
        env = GymBoard(random_seed=random_seed)
        games_nn.append(OrderedDict([("random_seed", env.random_seed)]))
        if log_steps:
            games_nn[i].update(OrderedDict([("steps", [])]))

        while True:
            prev_score = env.score
            observation = np.array([env.normalized_matrix])

            if use_eps_greedy_policy:
                actions = model.predict(observation, batch_size=1, verbose=0)

                if np.random.uniform() < eps:
                    action = np.random.random_integers(0, nb_actions - 1)
                else:
                    action = np.argmax(actions)
                action_status = env.move(action)
            else:
                action = model.predict_classes(observation, batch_size=1, verbose=0)[0]
                action_status = env.move(action)
                # help NN when it's stuck in invalid move
                if not action_status:
                    if log_steps:
                        games_nn[i]["steps"].append(OrderedDict([
                            ("action_name", actions[action]),
                            ("step_score", 0),
                            ("env_score", int(env.score)),
                            ("valid_move", False)
                        ]))

                    r = np.random.RandomState()
                    while True:
                        # random select from valid actions (remove invalid action predicted by NN or again with random choice)
                        # count in logging the first valid random action as valid move
                        valid_actions = actions.copy()
                        del valid_actions[action]
                        action = r.choice(valid_actions)
                        action_status = env.move(action)
                        if action_status:
                            break

            if log_steps and action_status:
                games_nn[i]["steps"].append(OrderedDict([
                    ("action_name", actions[action]),
                    ("step_score", int(env.score - prev_score)),
                    ("env_score", int(env.score)),
                    ("valid_move", True)
                ]))
            elif log_steps and not action_status:
                games_nn[i]["steps"].append(OrderedDict([
                    ("action_name", actions[action]),
                    ("step_score", 0),
                    ("env_score", int(env.score)),
                    ("valid_move", False)
                ]))

            if env.is_gameover():
                break

        games_nn[i].update(OrderedDict([("episode_score", env.score),
                                        ("highest_value", env.highest_value),
                                        ("nb_env_steps", env.n_steps_valid + env.n_steps_invalid),
                                        ("nb_env_steps_valid", env.n_steps_valid),
                                        ("nb_env_steps_invalid", env.n_steps_invalid)]))

        # random play
        env = GymBoard(random_seed=games_nn[i]["random_seed"])
        games_random.append(OrderedDict([("random_seed", env.random_seed)]))
        if log_steps:
            games_random[i].update(OrderedDict([("steps", [])]))

        r = np.random.RandomState()
        while True:
            prev_score = env.score
            action = r.choice(actions, 1)[0]
            action_status = env.move(action)

            if log_steps:
                games_random[i]["steps"].append(OrderedDict([
                    ("action_name", actions[action]),
                    ("step_score", int(env.score - prev_score)),
                    ("env_score", int(env.score)),
                    ("valid_move", True)
                ]))
            if env.is_gameover():
                break

        games_random[i].update(OrderedDict([("episode_score", env.score),
                                ("highest_value", env.highest_value),
                                ("nb_env_steps", env.n_steps_valid + env.n_steps_invalid),
                                ("nb_env_steps_valid", env.n_steps_valid),
                                ("nb_env_steps_invalid", env.n_steps_invalid)]))

    return pd.DataFrame(games_nn), pd.DataFrame(games_random)


def save_on_exit():
    print("Calling save_on_exit()...")

    global evaluation_results_nn
    global evaluation_results_random

    evaluation_results_nn = pd.concat(evaluation_results_nn, keys=keys)
    pd.to_pickle(evaluation_results_nn, evaluation_results_file.format(type="nn"))

    evaluation_results_random = pd.concat(evaluation_results_random, keys=keys)
    pd.to_pickle(evaluation_results_random, evaluation_results_file.format(type="random"))


if __name__ == "__main__":
    global evaluation_results_nn
    global evaluation_results_random

    ### OPTIONS ################################################################
    ### NN parameters
    n_games = int(1e5)
    n_epochs = 100
    n_hidden = 256
    batch_size = 128
    nb_actions = 4
    n_test_games = 100
    # play test games after fitting on 'n' games
    evaluate_after_n_games = 100

    # Keras NN model
    model = Sequential([
        Flatten(input_shape=(4, 4)),
        Dense(n_hidden),
        Activation("relu"),
        Dense(n_hidden),
        Activation("relu"),
        Dense(nb_actions),
        Activation("sigmoid")
    ])

    ### Other options
    # if True, stop the fitting when all game files are used
    # if False, you can have running generate_games_parallel.py in background and wait for the newly generated games
    stop_after = True
    # sleep for 'n' seconds until new games append
    sleep_time = 60
    verbose = 0
    register_save_on_exit = False

    if register_save_on_exit:
        atexit.register(save_on_exit)

    ### Data
    #root_dir = "/storage/brno2/home/jirinovo/_school/mvi/2048"
    root_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = model_name = "supervised-{n_epochs}_epochs-{n_games}k_games-{n_hidden}_hidden-512_min_score".format(
        n_epochs=n_epochs,
        n_games=round(n_games / 1000),
        n_hidden=n_hidden)
    model_file = "{root_dir}/models/{base_dir}/{model_name}-{{i_game}}.hd5".format(root_dir=root_dir,
                                                                                      base_dir=base_dir,
                                                                                      model_name=model_name)
    history_file = "{root_dir}/results/{base_dir}/history-{model_name}-{{i_game}}.pickle".format(root_dir=root_dir,
                                                                                                    base_dir=base_dir,
                                                                                                    model_name=model_name)
    evaluation_file = "{root_dir}/results/{base_dir}/evaluation-{{type}}-{model_name}-{{i_game}}.pickle".format(
        root_dir=root_dir,
        base_dir=base_dir,
        model_name=model_name)
    evaluation_results_file = "{root_dir}/results/{base_dir}/evaluation_complete-{model_name}.pickle".format(
        root_dir=root_dir,
        base_dir=base_dir,
        model_name=model_name)
    evaluation_results_best_file = "{root_dir}/results/{base_dir}/evaluation_best-{model_name}.pickle".format(
        root_dir=root_dir,
        base_dir=base_dir,
        model_name=model_name)
    epoch_results_best_file = "{root_dir}/results/{base_dir}/epochs_evaluation_best-{model_name}.pickle".format(
        root_dir=root_dir,
        base_dir=base_dir,
        model_name=model_name)
    data_path = "{root_dir}/data/random_games-50s_chunks-0_min_score-512_min_tile_value".format(root_dir=root_dir)
    ############################################################################

    try:
        os.mkdir("{root_dir}/models/{base_dir}".format(root_dir=root_dir, base_dir=base_dir))
    except FileExistsError:
        pass

    try:
        os.mkdir("{root_dir}/results/{base_dir}".format(root_dir=root_dir, base_dir=base_dir))
    except FileExistsError:
        pass

    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger()

    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose output.")
    else:
        logger.setLevel(logging.INFO)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy", "categorical_crossentropy"])

    epoch_results = []

    for i_epoch in range(n_epochs):
        logger.info("Epoch: {}".format(i_epoch + 1))
        i_games = 0
        i_games_current = 0
        i_observations = 0
        learned_files = []
        evaluation_results = []
        keys = []

        while True:
            game_file_list = [x for x in glob("{}/*".format(data_path)) if x not in learned_files]

            if not game_file_list:
                if stop_after:
                    logger.info("\tTrained on all available games. Stopping...")
                    break
                logger.info("\tWaiting for new generated games. Sleeping for {} seconds...".format(sleep_time))
                sleep(sleep_time)
                continue

            for game_file in game_file_list:
                logger.debug("\n\tGame file:", game_file)
                learned_files.append(game_file)
                with open(game_file, mode="rb") as f:
                    games = pickle.load(f)

                logger.debug("\t\tn games:", len(games))

                for i, game in enumerate(games):
                    X = np.array([step.matrix for step in game.steps])
                    y = np.array([step.action_encoded for step in game.steps])

                    n_observations = len(y)
                    i_observations += n_observations

                    logger.debug("\t\tn observations:", n_observations)

                    history = model.fit(X, y, batch_size=batch_size, nb_epoch=1, shuffle=False, verbose=verbose)

                    i_games_current += 1
                    i_games += 1

                    logger.debug("\t\tgame {}/{} done.".format(i+1, len(games)))

                    if i_games_current > evaluate_after_n_games:
                        model.save(model_file.format(i_game=i_games))
                        games_nn, games_random = evaluate(model, n_test_games)
                        i_samples_current = 0
                        i_games_current = 0
                        keys.append((i_games, i_observations))
                        evaluation_results.append(pd.concat([games_nn, games_random], axis=1, keys=["nn", "random"]))

                        logger.info(
                            "\n\tEVALUATION (NN model vs. random play), fitted on {} games ({} observations), playing {} games".format(
                                i_games,
                                i_observations,
                                n_test_games))
                        logger.info("\t\tMEAN EPISODE SCORE: {:.0f} | {:.0f}".format(games_nn["episode_score"].mean(),
                                                                                   games_random["episode_score"].mean()))
                        logger.info(
                            "\t\tMAX EPISODE SCORE: {} | {}".format(games_nn["episode_score"].max(),
                                                                  games_random["episode_score"].max()))
                        logger.info(
                            "\t\tMAX HIGHEST VALUE: {} | {}".format(games_nn["highest_value"].max(),
                                                                  games_random["highest_value"].max()))
                        logger.info("\t\tMEAN INVALID STEPS: {:.0f} | {:.0f}".format(games_nn["nb_env_steps_invalid"].mean(),
                                                                                   games_random["nb_env_steps_invalid"].mean()))

                logger.info("\t{}/{} games fitted.".format(i_games, n_games))

                if i_games > n_games:
                    model.save(model_file.format(i_game=i_games))
                    games_nn, games_random = evaluate(model, n_test_games)
                    keys.append((i_games, i_observations))
                    evaluation_results.append(pd.concat([games_nn, games_random], axis=1, keys=["nn", "random"]))
                    break

            if i_games > n_games:
                logger.info("\n\tFitted games ({}) > n_games ({}). Stopping and saving results...".format(i_games, n_games))
                break

        evaluation_results = pd.concat(evaluation_results, keys=keys)#, names=["n_games"])
        pd.to_pickle(evaluation_results, evaluation_results_file)
        evaluation_results_best = []
        for idx, df_select in evaluation_results.groupby(level=[0, 1]):
            df_nn = df_select["nn"]
            df_random = df_select["random"]

            evaluation_results_best.append(pd.concat([
                pd.DataFrame(OrderedDict([
                    ("n_games", idx[0]),
                    ("n_observations", idx[1])
                ]), index=[0]),

                pd.DataFrame(OrderedDict([
                    ("mean_episode_score", df_nn["episode_score"].mean()),
                    ("max_episode_score", df_nn["episode_score"].max()),

                    ("mean_highest_value", df_nn["highest_value"].mean()),
                    ("max_highest_value", df_nn["highest_value"].max()),

                    ("mean_nb_env_steps", df_nn["nb_env_steps"].mean()),
                    ("max_nb_env_steps", df_nn["nb_env_steps"].max()),

                    ("mean_nb_env_steps_valid", df_nn["nb_env_steps_valid"].mean()),
                    ("mean_nb_env_steps_valid", df_nn["nb_env_steps_valid"].max()),

                    ("mean_nb_env_steps_invalid", df_nn["nb_env_steps_invalid"].mean()),
                    ("max_nb_env_steps_invalid", df_nn["nb_env_steps_invalid"].max())
                ]), index=[0]),

                pd.DataFrame(OrderedDict([
                    ("mean_episode_score", df_random["episode_score"].mean()),
                    ("max_episode_score", df_random["episode_score"].max()),

                    ("mean_highest_value", df_random["highest_value"].mean()),
                    ("max_highest_value", df_random["highest_value"].max()),

                    ("mean_nb_env_steps", df_random["nb_env_steps"].mean()),
                    ("max_nb_env_steps", df_random["nb_env_steps"].max()),

                    ("mean_nb_env_steps_valid", df_random["nb_env_steps_valid"].mean()),
                    ("mean_nb_env_steps_valid", df_random["nb_env_steps_valid"].max()),

                    ("mean_nb_env_steps_invalid", df_random["nb_env_steps_invalid"].mean()),
                    ("max_nb_env_steps_invalid", df_random["nb_env_steps_invalid"].max())
                ]), index=[0]),
            ], axis=1, keys=["n_games", "nn", "random"]))

        evaluation_results_best = pd.concat(evaluation_results_best)
        evaluation_results_best.reset_index(drop=True)
        epoch_results.append(evaluation_results_best)
        #pd.to_pickle(evaluation_results_best, evaluation_results_best_file)

    epoch_results = pd.concat(epoch_results, keys=list(range(n_epochs)))
    pd.to_pickle(epoch_results, epoch_results_best_file)
    print("Trained on {} games ({} observations).".format(i_games, i_observations))
    print("\nepoch_results")
    print(epoch_results)
