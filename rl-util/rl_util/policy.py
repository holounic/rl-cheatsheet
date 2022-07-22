import pandas as pd
import random
from rl_util.common import S, A, P


class Policy:
    def p(self, a, s):
        raise NotImplementedError()


class StochasticPolicy(Policy):
    def __init__(self, state_space: int, action_space: int):
        self.p = pd.DataFrame({S: [], A: [], P: []})

        for state in range(state_space):
            p = [random.uniform(0, 1) for _ in range(action_space)]
            norm_factor = sum(p)
            for action in range(action_space):
                self.p = self.p.append({S: state, A: action, P: p[action] / norm_factor}, ignore_index=True)

    def p(self, a, s):
        return self.p.loc[(self.p[A] == a) & (self.p[S] == s)].sum()

    def __call__(self, s):
        i = self.p.loc[self.p[S] == s][P].idxmax()
        return self.p.iloc[i][A]

    def update(self, s, a, p):
        self.p.loc[(self.p[S] == s) & self.p[A] == a, P] = p


class DeterministicPolicy(Policy):
    def __init__(self, state_space, action_space=None):
        if action_space is None:
            self.states = [0 for _ in range(state_space)]
        else:
            self.states = [random.randint(0, action_space - 1) for _ in range(state_space)]

    def update(self, s, a):
        self.states[s] = a

    def p(self, a, s):
        return 1. if self.states[s] == a else 0

    def __call__(self, s):
        return self.states[s]


class EpsSoftPolicy(StochasticPolicy):
    def __init__(self, state_space: int, action_space: int, eps: float):
        super().__init__(state_space, action_space)
        self.eps = eps

    def update(self, s, a):
        n_actions = len(self.p.loc[self.p[s] == s])
        self.p.loc[self.p[S] == s, P] = self.eps / n_actions
        self.p.loc[(self.p[S] == s) & self.p[A] == a, P] = 1 - self.eps + self.eps / n_actions
