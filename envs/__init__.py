from gymnasium.envs.registration import register

register(
    id='SimpleActionOnly-v0',
    entry_point='gym_envs.simple_envs:SimpleActionOnlyEnv',
    kwargs={}
)

register(
    id='SimpleLargeAction-v0',
    entry_point='gym_envs.simple_envs:SimpleLargeActionEnv',
    kwargs={}
)

register(
    id='SimpleTwoStates-v0',
    entry_point='gym_envs.simple_envs:SimpleTwoStatesEnv',
    kwargs={}
)

register(
    id='SimpleSequence-v0',
    entry_point='gym_envs.simple_envs:SimpleSequenceEnv',
    kwargs={}
)

register(
    id='SimpleGoal-v0',
    entry_point='gym_envs.simple_envs:SimpleGoalEnv',
    kwargs={}
)
