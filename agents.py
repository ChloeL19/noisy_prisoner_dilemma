import random

class AutomatonAgent():
    def __init__(self, strategy):
        '''
        Arguments:
        - strategy: dictionary
            {
                state0: {'prob':p, 'CC': cc_state, 'CD': cd_state, 'DC': dc_state, 'DD': dd_state}
                ...
                ...
                ...
            }
        - MAYBE UNNECESSARY: initial_assumption: int, indicator of whether opponent in in 
            a cooperation mood or a defection mood
        Variables:
        - number to action map
            1:cooperate
            0:defect
        - curr_state
            holds the index of the current state for future reference
        '''
        self.strategy = strategy
        self.num_to_action_map = {1: 'C', 0: 'D'} # FIXME: bad sign we have this again in agent
        self.curr_state = 0 # this tracks which state of the strategy we are in

    # FIXME: how to make this "simultaneous" thing work??
    # especially with first action
    def get_action(self, end=False):
        '''
        - my_prev_action: int, action this agent took in previous timestep
        - other_prev_action: int, action other agent took in previous timestep
        - end: bool, whether or not we are in the last timestep of the episode
        (i.e. is this timestep 100?)
        - initial_assumption: None or int --> if not None, it's what we are assuming
            about other player is likely going to do
            1 -> they have a history of cooperation that implies further cooperation
            0 -> they have a history of defection that implies further defection
        '''
        action = 1 if random.random() <= self.strategy[self.curr_state]['prob'] else 0
        if end: # POTENTIAL FIXME
            self.curr_state = -1 # POSSIBLY FIXME: set to a nonsensical value because we are done playing
        return action
    
    # separate function for updating state
    def update_state(self, prev_action_profile=None):
        '''
        This function updates the state of the agent based on the previous action profile.
        - prev_action_profile: string, 'CC', 'CD', 'DD', 'DC'
        '''
        self.curr_state = self.strategy[self.curr_state][prev_action_profile] # tracks previous moves

# one key question that will come up: how to represent the automaton form
# in the style of an RL agent?
# actually the automatons remind me of attention --> it's actually equivalent to
# an RL algorithm with a lookback window of 5, and certain states lead to other states

# okay nope I am wrong. this is actually a way of restricting the action space I think
# the thing that distinguishes a state is the probability of cooperating
# so the decision to move to a new state is made based both on the other agent's previous action,
# your previous action, and your current state
# so want RL agent to output a one-hot vector: for each pair of previous inputs and previous output action,
# choose a new probability with which to cooperate --> there can be a max of 5 probabilities to choose from

# nah, just has two possible inputs and decides when to cooperate

# which RL algo is going to be best? I'm going to start with Q-learning
# and show to professor and then modify from there

class RL_agent():
    def __init__():
        pass

