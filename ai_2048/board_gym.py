import numpy as np


class GymBoard(object):
    """
    Represents the board of 2048 game and implements the OpenAI Gym environment (class Env) used in keras-rl:
    https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py
    Internally the board is a NumPy 2D array (matrix).
    """

    POSSIBLE_ACTIONS = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
        0: "up",
        1: "down",
        2: "left",
        3: "right"
    }

    NB_ACTIONS = 4

    def __init__(self, height=4, width=4, norm_const=0, max_wrong_steps=-1, step_normalize_matrix=True,
                 imp_move_penalty=-1, random_seed=0, step_normalize_reward=True, linear_penalty=True,
                 zero_invalid_move_reward=True):
        """
        Parameters
        ----------
        height : int
            Height of gaming board.
        width : int
            Width of gaming board.
        norm_const : int or float
            If > 0, then divide the board values by this constant after conversion to linear scale with log2.
            See normalize_matrix() for more details.
        max_wrong_steps : int
            If > 0, game over when number of following invalid steps is greater then max_wrong_steps.
        step_normalize_matrix : bool
            Whether to normalize matrix to linear scale (i.e. log2 all tiles).
        imp_move_penalty : int or float
            Penalty for invalid move.
        random_seed : int
            Random seed to create RandomState object and use it for initial tile positions and getting 2 or 4 each step.
            If 0, random seed will be randomly chosen.
        step_normalize_reward : bool
            Whether to also normalize reward to linear scale with log2 (like tile values).
        linear_penalty : bool
            Whether to linearly increase penalty with each new invalid step.
            E.g. penalties with imp_move_penalty = -1 will be each following invalid step: -1, -2, -3, -4, -5 | valid step -> reset | -1, -2, -3, ...
        zero_invalid_move_reward : bool
            If True, invalid move has zero reward.
        """
        
        self.norm_const = norm_const
        self.max_wrong_steps = max_wrong_steps
        self.step_normalize_matrix = step_normalize_matrix
        self.step_normalize_reward = step_normalize_reward
        self.linear_penalty = linear_penalty
        self.imp_move_penalty = imp_move_penalty
        self.zero_invalid_move_reward = zero_invalid_move_reward
        self.reset(height=height, width=width, random_seed=random_seed)

    def reset(self, height=4, width=4, random_seed=0):
        """
        Reset the board (env) to initial state.
        """
    
        if not random_seed:
            self.random_seed = np.random.randint(0, 4294967294)
        else:
            self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self._matrix = self._get_init_matrix(height, width)
        self._normalized_matrix = self.normalize_matrix()
        self.last_random_tile_index = ()
        self.last_random_tile_value = 0
        self.n_steps_valid = 0
        self.n_steps_invalid = 0
        self.n_steps_invalid_current = 0
        self.score = 0
        self.highest_value = 0

        if self.step_normalize_matrix:
            return self._normalized_matrix
        else:
            return self._matrix

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Parameters
        ----------
        action : object
            An action provided by the environment.

        Returns
        -------
        observation : object
            Agent's observation of the current environment.
        reward : float
            Amount of reward returned after previous action.
        done : bool
            Whether the episode has ended, in which case further step() calls will return undefined results.
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        old_score = self.score
        old_matrix = self.normalized_matrix
        action_status = self.move(action)
        score_diff = self.score - old_score
        done = self.is_gameover()

        if not action_status:
            if not done and not self.zero_invalid_move_reward:
                if self.linear_penalty:
                    reward = self.imp_move_penalty * self.n_steps_invalid_current
                else:
                    reward = self.imp_move_penalty
                # game over when n invalid steps are performed (max_wrong_steps > -1)
                if self.max_wrong_steps != -1 and self.n_steps_invalid_current > self.max_wrong_steps:
                    done = True
            else:
                reward = 0
        else:
            reward = score_diff
            if self.step_normalize_reward and reward > 0:
                reward = np.log2(reward)

        info = {"observation_prev": old_matrix, "score": self.score, "score_diff": score_diff}

        if self.step_normalize_matrix:
            return self._normalized_matrix, reward, done, info
        else:
            return self._matrix, reward, done, info

    @property
    def matrix(self):
        return self._matrix.copy()

    @matrix.setter
    def matrix(self, value):
        self._matrix = value
        self._normalized_matrix = self.normalize_matrix()

    @property
    def normalized_matrix(self):
        return self._normalized_matrix.copy()

    @property
    def shape(self):
        """
        Returns
        -------
        tuple(int, int)
            Tuple of the gaming board shape, i.e. the Numpy 2D array dimensions -> (rows, columns)
        """

        return self._matrix.shape

    def normalize_matrix(self):
        """
        Normalize matrix to linear scale with log2.
        If self.norm_const, also divide the values with self.norm_const
        """
    
        norm_matrix = np.ma.array(self._matrix, mask=(self._matrix == 0))
        norm_matrix = np.array(np.ma.log2(norm_matrix))
        if self.norm_const:
            norm_matrix /= self.norm_const
        return norm_matrix

    def get_random_tile_value(self):
        """
        Based on probability, return random tile value.
        Probabilities are from http://stackoverflow.com/a/22892362/1928742
        
        Returns
        -------
        int
            2 with probability of ~0.9 or 4 with probability of ~0.1
        """

        r = self.random_state.rand()

        if r < 0.9:
            return 2
        else:
            return 4

    def _get_init_matrix(self, width, height):
        """
        Creates the gaming matrix with defined shape and two random initial tiles (2 or 4).
        
        Returns
        -------
        numpy.array
        """

        matrix = np.zeros(shape=(height, width), dtype=np.int32)
        max_index_height = height - 1
        max_index_width = width - 1
        get_random_index = lambda: (self.random_state.randint(0, max_index_height),
                                    self.random_state.randint(0, max_index_width))
        random_tile_indexes = [get_random_index() for _ in range(2)]

        while random_tile_indexes[0] == random_tile_indexes[1]:
            random_tile_indexes[1] = get_random_index()

        matrix[random_tile_indexes[0]] = self.get_random_tile_value()
        matrix[random_tile_indexes[1]] = self.get_random_tile_value()

        return matrix

    def _move_line(self, array):
        """
        Moves and merges the tiles on one line of the gaming board.
        1) Count how many zero (empty) tiles are there.
        2) Extract the non-zero tiles in the reverse order.
        3) Merge them, count arising zeros.
        4) Reverse them to original order.
        5) Add zeros beforem them.

        example:
        1) split to the zero and non-zero tiles: [2 4 4 4 0 0 2 0 0 0 2] = [0 0 0 0 0] + [2 4 4 4 2 2]
        2) reverse the non-zero tiles array: [2 4 4 4 2 2] -> [2 2 4 4 4 2]
        3) merge the non-zero tiles and count the arising zeros: [2 2 4 4 4 2] -> [4 8 4 2] + [0 0]
        4) reverse to the original order: [4 8 4 2] -> [2 4 8 4]
        5) combine with the original and 'merge' zeros: [0 0 0 0 0] + [0 0] + [2 4 8 4] = [0 0 0 0 0 0 0 2 4 8 4]

        result: [2 4 4 4 0 0 2 0 0 0 2] -> [0 0 0 0 0 0 0 2 4 8 4]

        Parameters
        ----------
        array : numpy.array
            1D array (vector) to be moved and merged

        Returns
        -------
        numpy.array
            Merged numpy array.
        
        Examples
        --------
        >>> list(_move_line(np.array([2, 4, 4, 4, 0, 0, 2, 0, 0, 0, 2])))
        [0, 0, 0, 0, 0, 0, 0, 2, 4, 8, 4]
        >>> list(_move_line(np.array([2, 0, 0, 8, 0, 0, 4, 0, 0, 0, 2])))
        [0, 0, 0, 0, 0, 0, 0, 2, 8, 4, 2]
        """

        zeros_count = len(array[array == 0])
        new_array = np.flipud(array[array > 0])
        merge_array = []

        i = 0
        merge_zeros_count = 0
        score = 0

        while i != new_array.shape[0]:
            if i+1 == new_array.shape[0]:
                merge_array.append(new_array[i])
            elif new_array[i] == new_array[i+1]:
                merge_array.append(2 * new_array[i])
                i += 1
                merge_zeros_count += 1
                score += merge_array[-1]
            else:
                merge_array.append(new_array[i])
            i += 1

        merge_array = np.flipud(merge_array)
        zeros = (zeros_count + merge_zeros_count) * [0]
        zeros.extend(merge_array)
        return np.array(zeros), score

    def move(self, action, dry=False):
        """
        Move the tiles to defined direction and insert random tile.
        It slices the matrix to rows or lines and send them ordered in the movement
        direction to the _move_line function.

        example:
        matrix = [
            [2,  2,  4, 2, 8],
            [16, 32, 2, 8, 8],
            [32, 2,  2, 0, 0],
            [0,  0,  0, 0, 0],
            [0,  0,  2, 2, 0]
        ]

        'up' slices are columns from bottom to up: [0, 0, 32, 16, 2], [0, 0, 2, 32, 2] etc.
        'right': slices are rows from left to right: [2,  2,  4, 2, 8], [16, 32, 2, 8, 8] etc.
        
        Parameters
        ----------
        direction : str
            Direction to move on: 'up', 'down', 'left' or 'right'.

        Returns
        -------
        bool
            True if move was succesful, False otherwise (e.g. invalid move was performed).
        """

        direction = self.POSSIBLE_ACTIONS[action]

        if direction not in self.POSSIBLE_ACTIONS:
            raise ValueError("Unknown direction to move. Possible directions are 'up', 'down', 'left', 'right'")

        old_matrix = np.copy(self._matrix)
        score = 0

        if direction in [0, "up"]:
            lines_cols = [np.flipud(self._matrix[:, i]) for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                new_slice, score_slice = self._move_line(line)
                self._matrix[:, i] = np.flipud(new_slice)
                score += score_slice
        elif direction in [1, "down"]:
            lines_cols = [self._matrix[:, i] for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                new_slice, score_slice = self._move_line(line)
                self._matrix[:, i] = new_slice
                score += score_slice
        elif direction in [2, "left"]:
            lines_rows = [np.flipud(self._matrix[i, :]) for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                new_slice, score_slice = self._move_line(line)
                self._matrix[i, :] = np.flipud(new_slice)
                score += score_slice
        elif direction in [3, "right"]:
            lines_rows = [self._matrix[i, :] for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                new_slice, score_slice = self._move_line(line)
                self._matrix[i, :] = new_slice
                score += score_slice

        if np.array_equal(old_matrix, self._matrix):
            self.n_steps_invalid_current += 1
            self.n_steps_invalid += 1
            return False
        else:
            self.n_steps_valid += 1
            self.insert_random_tile()
            if self.step_normalize_matrix:
                self._normalized_matrix = self.normalize_matrix()
            self.n_steps_invalid_current = 0
            self.highest_value = self._matrix.max()

            if not dry:
                self.score += score
            return True

    def insert_random_tile(self):
        zero_indexes = np.where(self._matrix == 0)
        zero_indexes = list(zip(zero_indexes[0], zero_indexes[1]))
        random_zero_index = zero_indexes[self.random_state.choice(range(len(zero_indexes)))]
        self._matrix[random_zero_index] = self.get_random_tile_value()
        self.last_random_tile_index = random_zero_index
        self.last_random_tile_value = self._matrix[random_zero_index]

    def is_full(self):
        """
        Returns
        -------
        bool
            True if board doesn't contain zero (empty) tiles, False otherwise.
        """

        return not bool(len(np.where(self._matrix == 0)[0]))

    def is_gameover(self):
        """
        Checks if there are possible moves and if not, the game is over.
        
        Returns
        -------
        bool
            True if game is over, False otherwise.
        """

        original_matrix = np.copy(self._matrix)

        if self.is_full():
            move_results = []
            for move in range(self.NB_ACTIONS):
                move_results.append(self.move(move, dry=True))
            self._matrix = original_matrix
            return not any(move_results)
        else:
            return False

    def __repr__(self):
        return str(self.matrix)


# Some testing stuff.
if __name__ == "__main__":
    env = GymBoard(random_seed=2)
    print(env)
    r = np.random.RandomState(seed=1)
    from pprint import pprint
    actions = [0, 1, 2, 3]
    to_over = True

    if to_over:
        while True:
            pprint(env.matrix)
            print("SCORE:", env.score)
            print()
            action = r.choice(actions, 1)[0]
            env.move(action)
            print(env.POSSIBLE_ACTIONS[action])
            print()

            if env.is_gameover():
                print(env.matrix)
                print("GAME OVER!")
                break
    else:
        env.matrix = np.array([
            [2, 2, 0, 0],
            [2, 8, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        """
        env.matrix = np.array([
            [8,  2,  8,  4],
            [2, 64,  4,  8],
            [4, 16,  8,  4],
            [0,  2,  8,  2]])
        """

        print(env.matrix)
        print(env.score)

        print()
        env.move("right")
        print(env.matrix)
        print(env.score)

        """
        print()
        env.move("right")
        print(env.matrix)
        print(env.score)

        print()
        env.move("right")
        print(env.matrix)
        print(env.score)
        """