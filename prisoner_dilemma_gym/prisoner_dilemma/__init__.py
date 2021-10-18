from gym.envs.registration import register

register(
    id='prisoner-dilemma-v0',
    entry_point='prisoner_dilemma.envs:PrisonerDilemma',
)