import random
import pandas as pd
from rl_util.common import S, A, R, NS, P


class MarkovEnv:
    def __init__(self, start: set, terminal: set, action_space: int, state_space: int):
        self.start = start
        self.terminal = terminal
        self.transitions = pd.DataFrame({S: [], A: [], R: [], NS: [], P: []})
        self.state = None
        self.__action_space = action_space
        self.__state_space = state_space

    def state_space(self):
        return self.__state_space

    def action_space(self):
        return self.__action_space

    def add_state(self, state, action, reward, next_state, probability=1.):
        if state >= self.state_space() or action >= self.action_space() or next_state >= self.state_space():
            raise Exception(f'Out of bounds: state space is {self.state_space()}, action space is {self.action_space()}')
        self.transitions = self.transitions.append({S: state, A: action, R: reward, NS: next_state, P: probability}, ignore_index=True)

    def reset(self):
        self.state = random.choice(list(self.start))
        return self.state

    def step(self, action):
        if self.state in self.terminal:
            return None, None, True
        from_state = self.transitions.loc[(self.transitions[S] == self.state) & (self.transitions[A] == action)]
        next_states, probabilities, rewards = from_state[NS], from_state[P], from_state[R]
        i = random.choices(range(len(probabilities)), probabilities, k=1)[0]
        next_state, reward = next_states.iloc[i], rewards.iloc[i]
        self.state = next_state
        return next_state, reward, (self.state in self.terminal)
    
    def states(self):
        return self.transitions[S].unique()

    def p(self, state, action, reward, next_state):
        result = self.transitions.loc[
            (self.transitions[S] == state) &
            (self.transitions[A] == action) &
            (self.transitions[R] == reward) &
            (self.transitions[NS] == next_state)]
        return 0 if len(result) == 0 else result[P].sum()
