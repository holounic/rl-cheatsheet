def test_policy(env, policy, max_steps=10):
    total_reward = 0
    done = False
    steps = 0
    state = env.reset()
    trajectory = [state]
    while not done and steps <= max_steps:
        action = policy(state)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        trajectory.append(state)
    print(f'Finished in {steps} steps, reward: {total_reward}')
    return trajectory