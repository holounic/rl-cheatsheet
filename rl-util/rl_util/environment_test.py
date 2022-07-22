from rl_util.environment import MarkovEnv

if __name__ == '__main__':
    state_space = 4
    action_space = 2
    env = MarkovEnv(start={0}, terminal={3}, state_space=state_space, action_space=action_space)
    env.add_state(state=0, action=0, reward=1, next_state=1, probability=1.)
    env.add_state(state=0, action=1, reward=-1.5, next_state=2, probability=1.)

    env.add_state(state=1, action=0, reward=-1, next_state=2, probability=1.)
    env.add_state(state=1, action=1, reward=-1, next_state=3, probability=1.)

    env.add_state(state=2, action=0, reward=-1, next_state=3, probability=1.)
    env.add_state(state=2, action=1, reward=-.8, next_state=0, probability=1.)

    state = env.reset()
    N_STEPS = 5
    for _ in range(N_STEPS):
        print(env.step(1))