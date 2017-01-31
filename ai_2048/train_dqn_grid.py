"""
Train the agents for the game 2048. Perform grid search of
hyper-parameters. Each trained agent will be tested and compared to random play,
where both have env with same random seed. Report is provided as pandas DataFrame,
with statistics like score, maximum tile value, mean score, etc.

Vocabulary
  env ... game environment
  agent ... the piece of code which is behaving in env (i.e. playing the game)
  episode ... one game from beggining to game over
  
Reading
  https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
  Articles mentioned at https://github.com/matthiasplappert/keras-rl
"""

from board_gym import GymBoard
from callbacks import Logger2048
from util import save_history, get_callback, play_random_game

from keras.layers import Dense, GRU, Activation, Embedding, Reshape, Input, LSTM, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.callbacks import History

from rl.callbacks import TrainEpisodeLogger, TestLogger
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

import numpy as np
import pickle
import os
from itertools import product
import gc
import pandas as pd
import atexit
from collections import OrderedDict
from glob import glob


def save_on_exit():
    """
    Save the test result on exit signal (SIGTERM and so on).
    """

    print("Calling save_on_exit()...")

    global test_results
    global test_results_best

    test_results = pd.concat(test_results, keys=keys, names=key_parameter_names + ["i_test"])
    pd.to_pickle(test_results, test_results_file)


if __name__ == "__main__":
    # Register a function which will save test results on premature script end.
    atexit.register(save_on_exit)
    global test_results
    global test_results_best

    ### Reinforcement learning parameters
    nb_steps = int(2e6)
    nb_steps_annealed = int(1.25e6)
    nb_steps_warmup = int(0.25e6)
    nb_actions = 4
    train_interval = 100
    # Experience Replay memory size
    memory_size = int(2e6)
    # This is used for learning from raw screen, where some screen images are
    # joint together. Not for us, we are using direct env observations in form
    # of NumPy matrices.
    window_length = 1
    
    ### Hyper-parameters to search. For each product of these will be created and tested one model.
    # After how many steps update the target network. Values < 1 are using something called "softupdates".
    # See https://github.com/matthiasplappert/keras-rl/issues/55#issuecomment-265790717 for more info.
    target_model_update_list = [1e-3, 100, 1000, 10000]
    # gamma is the factor in discounted future reward. The lesser, the more
    # is future reward more uncertain, thanks to stochastic env.
    gamma_list = [0.8, 0.85, 0.9, 0.95, 0.99]
    # Policy = rules to select one action according to Q-values.
    # Boltzmann Q policy: http://computing.dcu.ie/~humphrys/Notes/RL/control.policy.html
    # Epsilon greedy policy: select random action with probability epsilon, otherwise select action with highest Q-value.
    # LinearAnnealedPolicy: linearly decrease the given policy parameters, so it gradually goes from exploration to exploitation.
    policy_list = [(BoltzmannQPolicy, "boltzmann", {}),
                   (EpsGreedyQPolicy, "eps_greedy", {}),
                   (LinearAnnealedPolicy, "annealed_boltzmann", {
                       "inner_policy": BoltzmannQPolicy,
                       "attr": "tau",
                       "value_max": 1.0,
                       "value_min": 0.1,
                       "value_test": 0.3,
                       "nb_steps": nb_steps_annealed
                   }),
                   (LinearAnnealedPolicy, "annealed_eps_greedy", {
                       "inner_policy": EpsGreedyQPolicy,
                       "attr": "eps",
                       "value_max": 1.0,
                       "value_min": 0.1,
                       "value_test": 0.05,
                       "nb_steps": nb_steps_annealed
                   })
    ]

    ### NN parameters
    nb_hidden = 256
    batch_size = 256
    # Some error clipping (was recommended by keras-rl author).
    delta_clip = 1.0
    n_test_episodes = 100

    ### Other options
    # Whether to use models trained in past, if they have the same hyper-parameters.
    use_previous = True
    verbose = 1
    key_parameter_names = ["target_model_update", "gamma", "policy"]
    key_model_names = ["dqn", "random"]

    ### Data
    # Root dir of project.
    root_dir = "/storage/plzen1/home/jirinovo/_school/mvi/2048"
    # You can use this to automatically get root dir, assuming it's the same dir as where is this script.
    #root_dir = os.path.dirname(os.path.realpath(__file__))
    # Base dir of models and results.
    base_dir = "grid_search-{n_steps_k}k_steps-{nb_hidden}_hidden-batch_size_{batch_size}-zero_inv_rew-V5".format(
        n_steps_k=nb_steps // 1000,
        nb_hidden=nb_hidden,
        batch_size=batch_size)
    # Base dir of some previous model.
    base_dir_prev = "grid_search-{n_steps_k}k_steps-{nb_hidden}_hidden-batch_size_{batch_size}-zero_inv_rew-V3".format(
        n_steps_k=nb_steps // 1000,
        nb_hidden=nb_hidden,
        batch_size=batch_size)
    # Model name where will be added hyper-parameter values.
    model_name_template = "dqn-{n_steps_k}k_steps-update_{{update}}-gamma_{{gamma}}-policy_{{policy}}".format(
        n_steps_k=nb_steps // 1000)
    # Path to model where will be added model name.
    model_file = "{root_dir}/models/{base_dir}/{{model_name}}-{{type}}.hd5".format(root_dir=root_dir, base_dir=base_dir)
    # Previous path to model.
    model_file_prev = "{root_dir}/models/{base_dir}/{{model_name}}-{{type}}.hd5".format(root_dir=root_dir, base_dir=base_dir_prev)
    # Path to save my pickled logger.
    logger_file = "{root_dir}/results/{base_dir}/logger-{{type}}-{{model_name}}.pickle".format(root_dir=root_dir,
                                                                                               base_dir=base_dir)
    # Path to save logger from keras-rl.
    logger_rl_file = "{root_dir}/results/{base_dir}/logger_rl-{{type}}-{{model_name}}.pickle".format(root_dir=root_dir,
                                                                                                     base_dir=base_dir)
    # Path to save test results.
    test_results_file = "{root_dir}/results/{base_dir}/test_results-pandas.pickle".format(root_dir=root_dir,
                                                                                          base_dir=base_dir)
    # Path to save best test results.
    test_results_best_file = "{root_dir}/results/{base_dir}/test_results_best-pandas.pickle".format(root_dir=root_dir,
                                                                                          base_dir=base_dir)

    # Make dirs for models and results.
    try:
        os.mkdir("{root_dir}/models/{base_dir}".format(root_dir=root_dir, base_dir=base_dir))
    except FileExistsError:
        pass

    try:
        os.mkdir("{root_dir}/results/{base_dir}".format(root_dir=root_dir, base_dir=base_dir))
    except FileExistsError:
        pass

    test_results = []
    keys = []
    existing_models = [x.split("/")[-1] for x in glob("{root_dir}/results/{base_dir}/logger-train-*".format(root_dir=root_dir,
                                                                                 base_dir=base_dir_prev))]
    logger_rl = None
    
    ### grid search of RL hyper-parameters
    for i, p in enumerate(product(target_model_update_list, gamma_list, policy_list)):
        target_model_update = p[0]
        gamma = p[1]
        policy_name = p[2][1]
        kwargs = p[2][2].copy()
        if kwargs:
            kwargs["inner_policy"] = kwargs["inner_policy"]()
        policy = p[2][0](**kwargs)

        model_name = model_name_template.format(update=target_model_update, gamma=gamma, policy=policy_name)

        print("Training model {}: target_model_update = {} | gamma = {} | policy = {}".format(
            i + 1,
            target_model_update,
            gamma,
            policy_name
        ))
        
        # 2048 environment
        env = GymBoard()

        # Keras NN model
        model = Sequential([
            Flatten(input_shape=(window_length, 4, 4)),
            Dense(nb_hidden),
            Activation("relu"),
            Dense(nb_hidden),
            Activation("relu"),
            Dense(nb_actions),
            Activation("linear")
        ])

        memory = SequentialMemory(limit=memory_size, window_length=window_length)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                       target_model_update=target_model_update, policy=policy, gamma=gamma, batch_size=batch_size,
                       train_interval=train_interval, enable_double_dqn=True, delta_clip=delta_clip)

        dqn.compile(Adam(), metrics=["mae"])
        
        # Whether to use existing model with the same hyper-parameters, skip its training and only do testing.
        if use_previous and logger_file.format(type="train", model_name=model_name).split("/")[-1] in existing_models:
            print("Model exists, skipping training...\n")
            dqn.load_weights(model_file_prev.format(model_name=model_name, type="weights"))
        else:
            callbacks = dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, callbacks=[Logger2048(verbose=verbose)])
            dqn.save_weights(model_file.format(model_name=model_name, type="weights"), overwrite=True)
            model.save(model_file.format(model_name=model_name, type="topology"))
            my_logger = get_callback(callbacks.callbacks, Logger2048)
            logger_rl = get_callback(callbacks.callbacks, TrainEpisodeLogger)
            save_history(my_logger, logger_file.format(type="train", model_name=model_name))
            save_history(logger_rl, logger_rl_file.format(type="train", model_name=model_name))
            print("Training done!")
        
        print("Testing...")

        dqn_tests = []
        random_tests = []
        for n in range(n_test_episodes):
            print("TEST EPISODE:", n + 1)
            env = GymBoard()
            callbacks = dqn.test(env, nb_episodes=1, visualize=False, verbose=2, callbacks=[Logger2048(verbose=verbose)])
            my_logger = get_callback(callbacks.callbacks, Logger2048)
            dqn_tests.append(my_logger.episodes[0])
            env = GymBoard(random_seed=env.random_seed)
            random_tests.append(play_random_game(env=env))

        dqn_tests = pd.DataFrame(dqn_tests)
        random_tests = pd.DataFrame(random_tests)
        test_results.append(pd.concat([dqn_tests, random_tests], axis=1, keys=key_model_names))
        keys.append((target_model_update, gamma, policy_name))

        # Some memory cleanup.
        del memory, dqn, callbacks, my_logger, model, env
        if logger_rl:
            del logger_rl
        gc.collect()

    test_results = pd.concat(test_results, keys=keys, names=key_parameter_names + ["i_test"])
    pd.to_pickle(test_results, test_results_file)

    test_results_best = []
    keys = []
    for idx, df_select in test_results.groupby(level=[0, 1, 2]):
        df_dqn = df_select["dqn"]
        df_random = df_select["random"]

        test_results_best.append(pd.concat([
            pd.DataFrame(OrderedDict([
                ("mean_episode_score", df_dqn["episode_score"].mean()),
                ("max_episode_score", df_dqn["episode_score"].max()),

                ("mean_highest_value", df_dqn["highest_value"].mean()),
                ("max_highest_value", df_dqn["highest_value"].max()),

                ("mean_nb_env_steps", df_dqn["nb_env_steps"].mean()),
                ("max_nb_env_steps", df_dqn["nb_env_steps"].max()),

                ("mean_nb_env_steps_valid", df_dqn["nb_env_steps_valid"].mean()),
                ("mean_nb_env_steps_valid", df_dqn["nb_env_steps_valid"].max()),

                ("mean_nb_env_steps_invalid", df_dqn["nb_env_steps_invalid"].mean()),
                ("max_nb_env_steps_invalid", df_dqn["nb_env_steps_invalid"].max())
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
        ], axis=1, keys=key_model_names))
        keys.append(idx)

    test_results_best = pd.concat(test_results_best, keys=keys, names=key_parameter_names)
    pd.to_pickle(test_results_best, test_results_best_file)

    print(test_results_best)
