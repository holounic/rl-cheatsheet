import random
from rl_util.environment import MarkovEnv


# Generates circle with random rewards, one start amd one terminal, all transitions have probability 1.0
def simple_circle(state_space: int = 4, action_space: int = 2):
    rewards = [-1, -2, -3]
    env = MarkovEnv(start={0}, terminal={state_space - 1}, action_space=action_space, state_space=state_space)
    for state in range(state_space - 1):
        env.add_state(state=state, action=0, reward=random.choice(rewards), next_state=state + 1, probability=1.)
        for action in range(1, action_space):
            env.add_state(
                state=state,
                action=action,
                reward=random.choice(rewards),
                next_state=random.randint(0, state_space - 1),
                probability=1.0)
    return env
