from rl.callbacks import Callback
from collections import OrderedDict


class Logger2048(Callback):
    """
    Specific logger for 2048 env. Saves env's score, highest value, etc.
    """

    POSSIBLE_ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, verbose=True):
        self.episodes = []
        self.verbose = verbose

    def on_episode_begin(self, episode, logs={}):
        self.current_episode = episode
        pass

    def on_episode_end(self, episode, logs={}):
        n_steps = self.env.n_steps_valid + self.env.n_steps_invalid

        if self.verbose:
            print("SCORE:", self.env.score,
                  "\tHIGHEST VALUE:", self.env.highest_value,
                  "\tVALID STEPS:", self.env.n_steps_valid,
                  "\tINVALID STEPS:", self.env.n_steps_invalid,
                  "\tTOTAL STEPS:", n_steps)

        self.episodes.append(OrderedDict([
            ("episode_score", self.env.score),
            ("highest_value", self.env.highest_value),
            ("nb_env_steps", n_steps),
            ("nb_env_steps_valid", self.env.n_steps_valid),
            ("nb_env_steps_invalid", self.env.n_steps_invalid),
            ("episode_reward", logs["episode_reward"]),
            ("random_seed", self.env.random_seed)
        ]))

    def on_step_begin(self, step, logs={}):
        pass

    def on_step_end(self, step, logs={}):
        pass

    def on_action_begin(self, action, logs={}):
        pass

    def on_action_end(self, action, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass
