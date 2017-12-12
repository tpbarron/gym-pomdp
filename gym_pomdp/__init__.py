from gym.envs.registration import register

# TMaze
register(
    id='TMaze-v0',
    entry_point='gym_pomdp.envs:TMazeEnv'
)
register(
    id='TMazeSimple-v0',
    entry_point='gym_pomdp.envs:TMazeSimpleEnv'
)
register(
    id='TMazeRacecar-v0',
    entry_point='gym_pomdp.envs:TMazeRacecarGymEnv'
)
