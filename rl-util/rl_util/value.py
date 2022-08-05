import pandas as pd
import numpy as np
from rl_util.common import S, A, V

class QFunction:
    def __init__(self, env):
        states, actions, values = [], [], []
        for state in range(env.state_space()):
            if env.is_terminal(state):
                continue
            for action in range(env.action_space()):
                states.append(state)
                actions.append(action)
                values.append(np.random.normal())
        self.q = pd.DataFrame({S: states, A: actions, V: values})

    def update(self, s, a, v):
        self.q.loc[(self.q[S] == s) & (self.q[A] == a), V] = v

    def __call__(self, s, a):
        result = self.q.loc[(self.q[S] == s) & (self.q[A] == a)]
        if len(result) == 0:
            return 0
        return result[V].values[0]

    def get_max(self, s):
        possible = self.q.loc[(self.q[S] == s)]
        if len(possible) == 0:
            return 0
        else:
            return possible[V].max()
