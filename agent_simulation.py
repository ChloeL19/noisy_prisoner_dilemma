import gym
from agents import AutomatonAgent 

random_agent = {
    0: {'prob': 0.5, 'CC':0,'CD':0,'DC':0,'DD':0}
}
agent1 = AutomatonAgent(strategy=random_agent)
agent2 = AutomatonAgent(strategy=random_agent)
env = gym.make('prisoner_dilemma:prisoner-dilemma-v0', enemy_agent = agent2)
action_a2 = agent2.get_action()
action_a1 = agent1.get_action()
state1, reward, done, info = env.step(action_a1)
print("state: {}, reward: {}, done: {}, info: {}".format(state1, reward, done, info))
# need to update the state of the second agent outside of the environment



# idea! make RL learn through an automaton agent. instead of random_agent, you pass
# in rl_strategy to the automaton agent --> this is not a static dictionary
# but instead an RL function that outputs the same things??? ehhh may not be possible
# oh I guess it is actually equivalent as I said before to an RL agent with look-back
# window of 1

# start by doing grid search of TfT probabilities

# what was the best strategy from class again????

# wondering how this will change if I try to pit two RL algorithms against
# each other

# which RL strategy most quickly learns an optimal strategy in this game?
    # look-back window for RL algorithm will only be one timestep
    # value function is (my last action, their last action) --> action
    # lets just make this a simple dense layer, see what it learns

# first step is seeing if agent can learn optimal strategy for taking advantage of 
# a fixed opponent
# idea: run RL against multiple different types of agents, see if it can generalize



# how to balance showcase of passion with showcase of new and exciting
# technical passion













# fascinating that we as a species forget how we learned to not forget through writing.
# but I digress from my code . . . 