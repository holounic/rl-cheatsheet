from rl_util.generator import simple_circle

if __name__ == '__main__':
    state_space = 4
    action_space = 2

    env = simple_circle(state_space, action_space)
    env.reset()

    for _ in range(10):

        env.step(1)