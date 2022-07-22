import jax.random as random
import random as std_random

class Markov:
    def __init__(self, transitions: dict, seed: int = 11):
        self.transitions = transitions
        self.state_space = len(self.transitions)
        self.transition_probs = {}
        self.rewards = {}

        key = random.PRNGKey(seed)
        for state in self.transitions.keys():
            self.rewards[state] = random.normal((key := random.split(key)[0]), (1,))

            self.transition_probs[state] = {}
            for action in self.transitions[state]:
                next_states = self.transitions[state][action]
                self.transition_probs[state][action] = {}
                state_probs = random.randint((key := random.split(key)[0]), (len(next_states),), 1,
                                             2 * len(next_states))
                state_probs = state_probs / state_probs.sum()
                for (next_state, state_prob) in zip(next_states, state_probs):
                    self.transition_probs[state][action][next_state] = state_prob

    def states(self):
        return list(self.transitions.keys())

    def state_space(self):
        return self.state_space

    def actions(self, state):
        return list(self.transition_probs[state].keys())

    def next_states(self, state):
        states = []
        for action in self.transition_probs[state].keys():
            states = states + list(self.transition_probs[state][action].keys())
        return list(set(states))

    def p(self, state, action, _, next_state):
        return self.transition_probs\
            .get(state, {action: {next_state: 0}})\
            .get(action, {next_state: 0})\
            .get(next_state, 0)

    def rewards(self, state):
        return self.rewards[state]


class StateInfo:
    def __init__(self, actions, next_states, rewards, probabilities):
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.probabilities = probabilities

    def choice(self, action):
        i = std_random.choices(range(len(self.next_states[action])), self.probabilities[action], k=1)[0]
        return self.next_states[action][i], self.rewards[action][i]

    def check_prob(self):
        for prob_for_action in self.probabilities.values():
            if not sum(prob_for_action) == 1.:
                return False
        return True

    def add_transition(self, action, next_state, reward, probability):
        self.actions.append(action)

        next_states = self.next_states.get(action, [])
        next_states.append(next_state)
        self.next_states[action] = next_states

        rewards = self.next_states.get(action, [])
        rewards.append(reward)
        self.rewards[action] = rewards

        probabilities = self.probabilities.get(action, [])
        probabilities.append(probability)
        self.probabilities[action] = probabilities


class MarkovEnv:
    def __init__(self):
        self.start = set()
        self.terminal = set()
        self.states = {}
        self.state = None

    def add_state(self, state, action, reward, next_state, probability):
        info = self.states.get(state, StateInfo([], {}, {}, {}))
        info.add_transition(action, next_state, reward, probability)
        self.states[state] = info

    def reset(self):
        self.state = std_random.choice(list(self.start))
        return self.state

    def step(self, action):
        info = self.states[self.state]
        next_state, reward = info.choice(action)
        self.state = next_state
        return next_state, reward, (reward in self.terminal)
    
    def states(self):
        return self.states.keys()
