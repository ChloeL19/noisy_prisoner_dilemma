import random
import tensorflow as tf
import numpy as np

class RL_agent():
    def __init__(self, initial_coop):
        '''
        Policy gradient approach. Or something like it.

        Variables
        - self.strategy: simple keras model for learning a function
            for outputing action to take
        - initial_coop: initial probability of cooperating
        - self.state: numpy array of arrays. Each element array has form
            [prob_of_cooeration, CC_ind, CD_ind, DC_ind, DD_ind]
            where each ind variable is a zero or one indicator variable
            we keep track of a history of 5 such arrays
        '''
        self.payoff = {
            'CC': (3,3),
            'CD': (0,5),
            'DC': (5,0),
            'DD': (0,0)
        }
        self.action_map = {
            0: 'D',
            1: 'C'
        }
        self.state = np.zeros((5,5))
        # initialize state with cooperative padding
        self.state[:, 0] = initial_coop
        self.state[:, 1] = 1
        #self.prev_coop = initial_coop
        #input_shape = [3] # previous coop prob, my previous action, other's prev (noisy) action
        input_shape = self.state.shape
        lr = 1e-6
        self.strategy = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1, input_shape = input_shape, activation='tanh'), # maybe try another layer
	    ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def compile_x(self, my_prev_action, other_prev_action):
        #x = np.asarray([self.prev_coop, my_prev_action, other_prev_action])
        # construct a new state: contain 5 history rounds
        x = np.expand_dims(self.state, axis=0)
        #assert(x.shape[1] == 3)
        return x
    
    def act():
        '''
        The same as get_action in old version but renamed to work with
        the simulation code.
        '''
        # if my_prev_action == None and other_prev_action == None:
        #     # randomly select an action here
        #     return 1 if random.random() <= 0.5 else 0

        x = self.compile_x(my_prev_action, other_prev_action)
        action = 1 if random.random() <= self.strategy.predict(x) else 0
        return action

    def update(self, my_prev_action, other_prev_action):
        '''
        After each round in the game, update the strategy based on the 
        observed rewards.
        - my_prev_action: int, my previous action
        - other_prev_action: int, other's noisy previous action
        '''
        coop_true_payoff = self.payoff['C' + self.action_map[other_prev_action]]
        def_true_payoff = self.payoff['D' + self.action_map[other_prev_action]]
        coop_noisy_payoff = self.payoff['C' + self.action_map[1 - other_prev_action]]
        def_noisy_payoff = self.payoff['D' + self.action_map[1 - other_prev_action]]
        with tf.GradientTape() as tape:
            # we cooperate if the cooperation probability is greater than 50%

            x = self.compile_x(my_prev_action, other_prev_action)
            coop_prob = self.strategy(x)
            # reward is expected payout for taking this action
            reward = -1*(0.95*(coop_prob*coop_true_payoff + (1-coop_prob)*def_true_payoff) +\
                 0.05*(coop_prob*coop_noisy_payoff + (1-coop_prob)*def_noisy_payoff))
            # have a sequence of state, action, reward
            # try to not change how this works from the previous project
            # don't explicitly account for the 5%

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