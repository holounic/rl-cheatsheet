from rl_util import EpsSoftPolicy

if __name__ == '__main__':
    policy = EpsSoftPolicy(4, 2, 0.1)
    policy.update(1, 1)
    policy(1)
    print(policy._p)