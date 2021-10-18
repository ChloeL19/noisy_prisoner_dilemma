import gym
import random

# To use this environment, be sure to type the following in terminal:
# pip install -e prisoner_dilemma_gym

class PrisonerDilemma(gym.Env):
    def __init__(self, enemy_agent=None, max_num_rounds=100, noise=0.05):
        '''
        Initialize the payoff matrix of the prisoner's dilemma game.
        CC: (3,3), CD: (0,5), DC: (5,0), DD:(0,0)
        Initialize the "enemy agent".
        General idea with this class:
        - players take steps simultaneously, so both must submit plan for step
        - the returned state or observation has noise
        - reward for timestep determined by payoff matrix
        - info contains information about who wins
        - number of rounds capped at 100
        - actions:
            - cooperate: 1
            - defect: 0
        - action profile: (0,0)

        *note: I think I incorrectly interchange "state" and "action profile"
        in a few places

        -----------arguments---------------
        - enemy_agent: part of the agent class
        '''
        if enemy_agent == None:
            raise Exception("Enemy agent should not be None!")
        else:
            self.enemy_agent=enemy_agent
        self.player_two = enemy_agent
        self.max_rounds = max_num_rounds
        self.curr_round = 0 # for tracking how many rounds we've been through
        self.noise = noise
        self.noisy_numeric_state_p1 = (-1, -1) # noisy from p1's perspective
        self.noisy_numeric_state_p2 = (-1, -1) # noisy from p2's perspective
        self.noisy_state_p1 = None # the string version of the previous two states
        self.noisy_state_p2 = None
        self.true_numeric_state = (-1, -1)
        self.true_state = None # string version of previous state
        self.done = False
        self.info = {}
        self.action_map = {
            0: 'D',
            1: 'C'
        }
        self.payoff = {
            'CC': (3,3),
            'CD': (0,5),
            'DC': (5,0),
            'DD': (0,0)
        }

    def step(self, action):
        '''
        Returns the updated state (the actions taken by each agent with noise).
        Returns the payoff (based on non-noisy actions)
        returns done indicator (True if max number of rounds reached)
        returns info (contains the running sum of payoff for each player)
        All tuples structured in the following way: (player1, player2)
        Arguments:
        - action: int, the action for player 1, either 0 or 1
            0 is defect and 1 is cooperate
        '''
        # calculate action for player 2 based on noisy state
        # going to attempt making the initial state (-1,-1), for RL training purposes
        # later --> maybe a bad idea
        if self.curr_round >= self.max_rounds:
            self.done = True

        # calculate the next action for the opposing agent
        p2_action = self.player_two.get_action(end=self.done) # feed in the noisy states

        # update our true state, should capture the numeric version
        self.true_numeric_state = (action, p2_action)
        self.true_state = self.action_map[action] + self.action_map[p2_action]
        # introduce noise to the noisy states, only change the "other" action
        self.noisy_numeric_state_p1 = (action, (1-p2_action) if random.random() <= self.noise else p2_action)
        self.noisy_numeric_state_p2 = ((1-action) if random.random() <= self.noise else action, p2_action)
        self.noisy_state_p1 = self.action_map[self.noisy_numeric_state_p1[0]] + self.action_map[self.noisy_numeric_state_p1[1]]
        self.noisy_state_p2 = self.action_map[self.noisy_numeric_state_p2[0]] + self.action_map[self.noisy_numeric_state_p2[1]]

        # update the internal state of the enemy agent
        self.enemy_agent.update_state(prev_action_profile=self.noisy_state_p2)
        # maybe should also update my current agent here lol

        # calculate the reward based on the non-noisy actions
        payoff_tuple = self.payoff[self.true_state]

        # update the info
        self.info['p1_noisy_action_profile'] = self.noisy_state_p1
        self.info['p2_noisy_action_profile'] = self.noisy_state_p2
        #self.info['payoff'] = payoff_tuple

        self.curr_round += 1
        # return (noisy) state for player 1, reward, done, info
        return self.noisy_state_p1, payoff_tuple, self.done, self.info

    def reset(self):
        self.curr_round = 0
        pass
        # return state, info (?)

    def render(self):
        # just print stuff?
        pass





## Microsoft Interview Notes
# growth mindset: how could I improve the situation?
# where is my passion? what is my story in a concise way?

# actual coding part: what would it be like to work with you?
# ask clarifying questions
# give sample input, make sure we both understand the problem
# make this more of a dialogue
# think through some alternative approaches, explain why the one I'm 
# choosing is the best one


# Data science:
# who are you in 1-2 minutes, have this practiced
# highlight things you did. what didn't work, what did you learn from that
# please ask questions --> make sure you are interested in company to which you are
# applying
    # what is there day-to-day life, what does it look like to work in your team?



# my notes:

# passionate about understanding things shaping our world
# ai with power dynamics, big shifts coming, how can I be there to help shape this






# MOCK INTERVIEW NOTES
# good amount detail, not too little, not too much --> showed the interest in that project, that I was able to do something that was
# meaningful
# talked about how working with others, rapid feedback
# how worked with manager: when is right time to talk with manager, good job answering
# shows: able to do work yourself but also know when to ask for help
    # there is a right time to ask for help
# important balance: want to try things myself, but also don't never ask for help
# demonstrate all of this

# technical question:
    # good job asking clarifying questions
    # i hadn't internalize the question correctly
        # kept asking questions to correct
    # try getting to code sooner --> it will force that conversation sooner
        # getting there earlier is valuable, they may or may not catch on early
    # good job confirming constraints, stating the things that I was thinking
    # once we clarified the diconnect, I had much better idea of what to do

    # honest questions come through really well
    # 4 or 5 interviews --> on interview 3, fine to say honestly no questions
        # i asked this question so-and-so this question, I would love to hear your
        # experience on that 


# hackerrank.com --> lots of interview questions --> 1 per day
# get used to typing stuff out