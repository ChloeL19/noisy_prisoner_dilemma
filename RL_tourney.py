import random
import tensorflow as tf
import keras
import numpy as np
import datetime
import os
import glob

class RL_agent():
    def __init__(self, initial_coop, test=False):
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
        self.name = "RL"
        self.current_state = 0 # for interfacing with other code, kind of hacky
        self.input_dim = 5
        self.state = np.random.rand(1,self.input_dim)

        # initialize state with cooperative padding
        # self.state[0, 0] = initial_coop
        # self.state[0, 1] = 1
        self.prev_coop = initial_coop

        # for updates
        self.cumu_loss = 0

        # for saving checkpoint models
        self.outdir = None

        # defining the model itself
        lr = 1e-6
        if test and os.listdir("./trained_models/") != []:
            # get the latest model saved in the training models directory
            list_of_files = glob.glob("./trained_models/*")
            self.outdir = max(list_of_files, key=os.path.getctime)
            self.strategy = self.load_model()
        else:
            print("Building a new model")
            self.strategy = tf.keras.Sequential()
            self.strategy.add(tf.keras.layers.Dense(32, input_dim = self.input_dim, activation='relu'))
            self.strategy.add(tf.keras.layers.Dense(2, activation = "softmax"))
            self.strategy.build()
            #optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
            #compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
            # self.strategy = tf.keras.models.Sequential([
            #     # maybe include a batch normalization layer
            #     tf.keras.layers.LSTM(units=10, input_shape=self.input_shape, #recurrent_dropout=0.1,
            #     activation='relu'),
            #     tf.keras.layers.Dense(units=1, input_shape = self.input_shape, activation='softmax'), # maybe try another layer
            # ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # for tracking gradients during training
        self.gradBuffer = self.strategy.trainable_variables
        for ix,grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

    def getStrategy(self):
        '''
        For compatibility with the tournament, include this function.
        '''
        return "RL agent strategy."
    
    def get_obs(self):
        '''
        Get input of the proper form for our RL algorithm.
        '''
        #return np.expand_dims(self.state, axis=0)
        return self.state.reshape([1,5])

    def get_action_prob(self):
        #x = self.get_obs()
        return self.strategy.predict(self.state) #.flatten()[0]
     
    def act(self, train=None):
        '''
        The same as get_action in old version but renamed to work with
        the simulation code.
        - train: bool, is this 
        '''
        logits = self.get_action_prob()

        # if coop_prob == 1.0:
        #     coop_prob -= 0.01
        # elif coop_prob == 0.0:
        #     coop_prob += 0.01
        # print("RL state: {}".format(self.state))
        # print("RL cooperation prob: {}".format(coop_prob))

        # action = "C" if random.random() <= coop_prob else "D"
        # self.prev_coop = coop_prob

        a_dist = logits
        # Choose random action with p = action dist
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)

        # idea: for testing maybe try just selecting the argmax
  
        return "C" if a == 0 else "D"

    def react(self, state_ind):
        '''
        Changes the state that will be fed into the RL agent.
        - state_ind: index of the indicator variable that needs
            to be toggled
        '''
        # update the state
        self.state = np.zeros((1, self.input_dim))
        self.state[0, 0] = self.prev_coop
        self.state[0, state_ind+1] = 1

        # self.state[1:, :] = self.state[:-1, :]
        # self.state[0, :] = np.zeros((5,))
        # self.state[0, state_ind+1] = 1
        # self.state[0, 0] = self.prev_coop

    def update(self, train=None, timestep_reward=None):
        '''
        After each round in the game, update the strategy based on the 
        observed rewards.

        Idea: treat this as a binary classification problem (kind of).
        Compute the binary cross-entropy between the ideal action and 
        the chosen action for each timestep
        Minimize the sum of this loss over timesteps
        Is this breaking something related to reinforcement learning??

        Nahhh I feel this is wrong.
        Instead compute the expected reward value given the action of
        the other player. 

        I guess this will become a model-based policy gradient algorithm.

        - train: bool, only do stuff if this is true and otherwise
            just pass without doing anything
        - opponent_action: char, the action (either cooperate or defect) taken
            by the opposing player
        - timestep_reward: score for current timestep
        '''
        if train:
            # need to MASSIVELY fix this
            with tf.GradientTape() as tape:
                x = self.get_obs()
                prob_coop = self.strategy(x)[0][0]
                if prob_coop == 1.0:
                    prob_coop -= 0.01
                if prob_coop == 0.0:
                    prob_coop += 0.01
                discount_factor = 0.95 
                
                timestep_loss = -tf.math.log(1-prob_coop) / (timestep_reward if timestep_reward > 0 else 0.1)
                self.cumu_loss = discount_factor*self.cumu_loss + timestep_loss


            grads = tape.gradient(self.cumu_loss, self.strategy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.strategy.trainable_variables))
            return self.cumu_loss
        else:
            return None

    def save_model(self):
        '''
        Save the current model so it can be re-loaded for
        future testing later.
        '''
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.outdir = "./trained_models/" + current_time
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        self.strategy.save(self.outdir)

    def load_model(self):
        '''
        Load the saved model for testing purposes.
        '''
        print("Loading model from {}\n".format(self.outdir))
        return keras.models.load_model(self.outdir)