import random
import tensorflow as tf
import numpy as np
from jax import grad
import jax.numpy as jnp

# helpful jax info: https://github.com/google/jax

class Simulation():
    random_agent = {
        0: {'prob': 0.5, 'CC':0,'CD':0,'DC':0,'DD':0}
    }
    tft_agent = {
        0: {'prob': 0.99, 'CC': 0, 'CD': 1, 'DC': 0, 'DD': 1},
        1: {'prob': 0.01, 'CC': 0, 'CD': 1, 'DC': 0, 'DD': 1}
    }
    action_map = {
        0: 'D',
        1: 'C'
    }
    payoff = {
        'CC': (3,3),
        'CD': (0,5),
        'DC': (5,0),
        'DD': (0,0)
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
        action = 1 if random.random() <= self.strategy[self.curr_state]['prob'] else 0
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
    def __init__(self, initial_coop):
        '''
        Use a SARSA approach for learning this strategy.
        Design and train a function that takes in previous probability of 
        cooperating and other player's action and outputs 
        new probability of cooperating.

        Actually I think I'm using a policy gradient approach here.

        Variables
        - self.strategy: simple keras model for learning a function
            for outputing probability of cooperating
        - initial_coop: initial probability of cooperating
        '''
        self.prev_coop = initial_coop
        input_shape = [3] # previous coop prob, my previous action, other's prev (noisy) action
        lr = 1e-6
        self.strategy = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1, input_shape = input_shape, activation='tanh'), # maybe try another layer
            # tf.keras.layers.BatchNormalization(),
            # consider adding a convolutional layer here
            #tf.keras.layers.Dense(units=1, activation='linear')
	    ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def strategy(self, x):
        '''
        Return a probability for cooperating based on my previous action,
        the other's previous action, and my previous cooperation probability.
        This should be differentiable in JAX.
        - x: array of size [1,3] with aforementioned information
        '''
        # do a bunch of matrix math
        pass

    def compile_x(self, my_prev_action, other_prev_action):
        x = np.asarray([self.prev_coop, my_prev_action, other_prev_action])
        x = np.expand_dims(x, axis=0)
        #assert(x.shape[1] == 3)
        return x

    def jax_compile_x(self, my_prev_action, other_prev_action):
        '''
        The same as compile_x but now with jax.
        '''
        x = jnp.asarray([self.prev_coop, my_prev_action, other_prev_action])
        x = jnp.expand_dims(x, axis=0)
        #assert(x.shape[1] == 3)
        return x

    def jax_update(self, my_prev_action, other_prev_action):
        '''
        Update the parameters in my strategy function with Jax.
        '''
        pass

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
            # I am actually very confused about how to update this
            # what am I maximizing here??
            # supposedly reward in the timestep

            # idea: given the previous action of the other player, 
            # define reward as the expected value probability_coop*val_coop +
            # prob_defect * val_defect
            # define loss as the negative of this
            # optimize this loss

            x = self.compile_x(my_prev_action, other_prev_action)
            coop_prob = self.strategy(x)
            # reward is expected payout for taking this action
            reward = -1*(0.95*(coop_prob*coop_true_payoff + (1-coop_prob)*def_true_payoff) +\
                 0.05*(coop_prob*coop_noisy_payoff + (1-coop_prob)*def_noisy_payoff))

        grads = tape.gradient(reward, self.strategy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.strategy.trainable_variables))
        # update the previous coop probability
        self.prev_coop = np.asarray(coop_prob)[0][0]

        # FIXME!!! Attempt this with Jax instead!!!


    def terminal_update(self):
        '''
        Terminal update that updates strategy based on total reward at the end.
        I'm not convinced it's worth having this function.
        Not convinced it's worth differentiating between a terminal 
        and non-terminal update. 
        '''
        pass

    def get_action(self, my_prev_action=None, other_prev_action=None):
        '''
        Return an int for cooperation (1) or defection (0)
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