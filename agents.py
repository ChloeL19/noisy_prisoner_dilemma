import random
import tensorflow as tf
import numpy as np

class Simulation():
    random_agent = {
        0: {'prob': 0.5, 'CC':0,'CD':0,'DC':0,'DD':0}
    }
    tft_agent = {
        0: {'prob': 0.99, 'CC': 0, 'CD': 1, 'DC': 0, 'DD': 1},
        1: {'prob': 0.01, 'CC': 0, 'CD': 1, 'DC': 0, 'DD': 1}
    }
    action_map = {
        0: 'C',
        1: 'D'
    }
    payoff = {
        'CC': (3,3),
        'CD': (0,5),
        'DC': (5,0),
        'DD': (1,1)
    }

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

        Variables:
        - number to action map
            1:cooperate
            0:defect
        - curr_state
            holds the index of the current state for future reference
        '''
        self.strategy = strategy
        self.curr_state = 0 # this tracks which state of the strategy we are in

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
        action = 0 if random.random() <= self.strategy[self.curr_state]['prob'] else 1
        if end: 
            self.curr_state = 0 
        return action
    
    # separate function for updating state
    def update_state(self, prev_action_profile=None):
        '''
        This function updates the state of the agent based on the previous action profile.
        - prev_action_profile: string, 'CC', 'CD', 'DD', 'DC'
        '''
        try:
            self.curr_state = self.strategy[self.curr_state][prev_action_profile] # tracks previous moves
        except:
            import pdb; pdb.set_trace();

class RL_agent():
    def __init__(self, initial_coop):
        '''
        Policy gradient approach. Or something like it.

        Variables
        - self.strategy: simple keras model for learning a function
            for outputing action to take
        - initial_coop: initial probability of cooperating
        '''
        self.prev_coop = initial_coop
        input_shape = [3] # previous coop prob, my previous action, other's prev (noisy) action
        lr = 1e-6
        self.strategy = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1, input_shape = input_shape, activation='tanh'), # maybe try another layer
	    ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    

    def compile_x(self, my_prev_action, other_prev_action):
        x = np.asarray([self.prev_coop, my_prev_action, other_prev_action])
        x = np.expand_dims(x, axis=0)
        #assert(x.shape[1] == 3)
        return x

    def update(self, my_prev_action, other_prev_action):
        '''
        After each round in the game, update the strategy based on the 
        observed rewards.
        - my_prev_action: int, my previous action
        - other_prev_action: int, other's noisy previous action
        '''
        sim = Simulation()
        coop_true_payoff = sim.payoff['C' + sim.action_map[other_prev_action]]
        def_true_payoff = sim.payoff['D' + sim.action_map[other_prev_action]]
        coop_noisy_payoff = sim.payoff['C' + sim.action_map[1 - other_prev_action]]
        def_noisy_payoff = sim.payoff['D' + sim.action_map[1 - other_prev_action]]
        with tf.GradientTape() as tape:
            # we cooperate if the cooperation probability is greater than 50%

            x = self.compile_x(my_prev_action, other_prev_action)
            coop_prob = self.strategy(x)
            # reward is expected payout for taking this action
            reward = -1*(0.95*(coop_prob*coop_true_payoff + (1-coop_prob)*def_true_payoff) +\
                 0.05*(coop_prob*coop_noisy_payoff + (1-coop_prob)*def_noisy_payoff))

        grads = tape.gradient(reward, self.strategy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.strategy.trainable_variables))
        # update the previous coop probability
        self.prev_coop = np.asarray(coop_prob)[0][0]

    def get_action(self, my_prev_action=None, other_prev_action=None):
        '''
        Return an int for cooperation (1) or defection (0)

        POSSIBLY FIX THE SAMPLING SCHEME HERE
        '''
        if my_prev_action == None and other_prev_action == None:
            # randomly select an action here
            return 1 if random.random() <= 0.5 else 0
        x = self.compile_x(my_prev_action, other_prev_action)
        action = 1 if random.random() <= self.strategy.predict(x) else 0
        return action

    def get_action_prob(self, my_prev_action, other_prev_action):
        return self.strategy(self.compile_x(my_prev_action,
        other_prev_action))