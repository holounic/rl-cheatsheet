import jax.random as random


class Policy:
    def p(self, a, s):
        raise NotImplementedError()


class StochasticPolicy(Policy):
    def __init__(self, state_actions: dict, seed: int = 11):
        key = random.PRNGKey(seed)
        self.probs = {}

        for state in state_actions:
            self.probs[state] = {}
            actions = state_actions[state]

            action_probs = random.randint((key := random.split(key)[0]), (len(actions),), 1, 2 * len(actions) + 1)
            action_probs = action_probs / action_probs.sum()

            for (action, prob) in zip(actions, action_probs):
                self.probs[state][action] = prob

    def p(self, a, s):
        return self.probs[s][a]

    def __call__(self, s):
        probs = self.probs[s]
        max_a = probs.keys()[0]
        for a in probs.keys():
            if probs[a] > probs[max_a]:
                max_a = a
        return max_a


class DeterministicPolicy(Policy):
    def __init__(self, transitions: dict):
        self.transitions = transitions

    def p(self, a, s):
        return 1. if self.transitions[s] == a else 0

    def __call__(self, s):
        return self.transitions[s]


class EpsSoftPolicy(Policy):
    def __init__(self, state_actions: dict, eps: float, seed: int = 11):
        self.eps = eps
        key = random.PRNGKey(seed)
        self.probs = {}

        for state in state_actions:
            self.probs[state] = {}
            actions = state_actions[state]

            action_probs = random.randint((key := random.split(key)[0]), (len(actions),), 1, 2 * len(actions) + 1)
            action_probs = action_probs / action_probs.sum()

            for (action, prob) in zip(actions, action_probs):
                self.probs[state][action] = prob

    def p(self, a, s):
        return self.probs[s][a]

    def __call__(self, s):
        probs = self.probs[s]
        max_a = probs.keys()[0]
        for a in probs.keys():
            if probs[a] > probs[max_a]:
                max_a = a
        return max_a

    def update(self, s, best_a):
        n_actions = len(self.probs[s])
        a_star_prob = 1 - self.eps + self.eps / n_actions
        other_prob = self.eps / n_actions

        for action in self.probs[s].keys():
            self.probs[s][action] = a_star_prob if action == best_a else other_prob
