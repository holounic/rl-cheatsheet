import pandas as pd
import random
from rl_util.common import S, A, P, V


class Policy:
    def __init__(self, state_space: int, action_space: int):
        self.state_space = state_space
        self.action_space = action_space

    def p(self, a, s):
        raise NotImplementedError()


class StochasticPolicy(Policy):
    def __init__(self, state_space: int, action_space: int):
        super().__init__(state_space, action_space)
        self._p = pd.DataFrame({S: [], A: [], P: []})

        p = 1 / action_space
        for state in range(state_space):
            for action in range(action_space):
                self._p = self._p.append({S: state, A: action, P: p}, ignore_index=True)

    def p(self, a, s):
        return self._p.loc[(self._p[A] == a) & (self._p[S] == s)][P].sum()

    def __call__(self, s):
        i = self._p.loc[self._p[S] == s][P].idxmax()
        return self._p.iloc[i][A]

    def update(self, s, a, p):
        self._p.loc[(self._p[S] == s) & self._p[A] == a, P] = p


class DeterministicPolicy(Policy):
    def __init__(self, state_space, action_space=None):
        super().__init__(state_space, action_space)
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
        self._p.loc[self._p[S] == s, P] = self.eps / self.action_space
        self._p.loc[(self._p[S] == s) & (self._p[A] == a), P] = 1 - self.eps + self.eps / self.action_space

    def __call__(self, s):
        p = self._p.loc[self._p[S] == s]
        return random.choices(p[A].values, p[P].values, k=1)[0]


class EpsSoftPolicyFromQ(EpsSoftPolicy):
    def __init__(self, q, state_space: int, action_space: int, eps: float):
        super().__init__(state_space, action_space, eps)
        self.q = q

    def update(self, s, a):
        raise Exception(':(')

    def p(self, a, s):
        q = self.q.loc[self.q[S] == s].reset_index()
        best_a = q[A].values[q[V].idxmax()].astype(int)
        if a == best_a:
            return 1 - self.eps + self.eps / self.action_space
        else:
            return self.eps / self.action_space

    def __call__(self, s):
        q = self.q.loc[self.q[S] == s].reset_index()
        if len(q) == 0:
            return random.randint(0, self.action_space - 1)
        a = q[A].values
        best_a = a[q[V].idxmax()].astype(int)
        probs = [self.eps / self.action_space for _ in range(self.action_space)]
        probs[best_a] = 1 - self.eps + self.eps / self.action_space
        return random.choices(list(range(self.action_space)), probs, k=1)[0]


class GreedyPolicyFromQ(DeterministicPolicy):
    def __init__(self, q, state_space: int, action_space: int):
        super().__init__(state_space, action_space)
        self.q = q

    def update(self, s, a):
        raise Exception(':(')

    def p(self, a, s):
        q = self.q.loc[self.q[S] == s].reset_index()
        best_a = q[A].values[q[V].idxmax()].astype(int)
        return 1 if a == best_a else 0

    def __call__(self, s):
        q = self.q.loc[self.q[S] == s].reset_index()
        if len(q) == 0:
            return random.randint(0, self.action_space - 1)
        a = q[A].values
        best_a = a[q[V].idxmax()].astype(int)
        return best_a