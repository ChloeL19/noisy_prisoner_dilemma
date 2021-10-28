import gym
from agents import AutomatonAgent, RL_agent, Simulation


################ Train the RL Agent ########################
sim1 = Simulation()
max_num_rounds = 200
done = False
env = gym.make('prisoner_dilemma:prisoner-dilemma-v0', 
                            enemy_agent=AutomatonAgent(strategy=sim1.random_agent),
                            action_map=sim1.action_map, 
                            payoff=sim1.payoff,
                            max_num_rounds=max_num_rounds)
rlagent = RL_agent(initial_coop=0.5)
a1 = rlagent.get_action()
train_rewards = []
total_r1 = 0
total_r2 = 0
while (not done):
    state, reward, done, info = env.step(a1)
    train_rewards.append(reward)
    rlagent.update(*state)
    a1 = rlagent.get_action(*state)

for (r1, r2) in train_rewards:
    total_r1 += r1
    total_r2 += r2
print("--------------------Total Training Reward------------------")
print("RL agent: {}, Other agent: {}".format(total_r1, total_r2))

################# Test the RL Agent #########################
env.reset()
done = False
test_rewards = []
total_r1 = 0
total_r2 = 0
a1 = 1 # assume always start by cooperating
while (not done):
    state, reward, done, info = env.step(a1)
    #print(state, reward, done, info)
    test_rewards.append(reward)
    a1 = rlagent.get_action(*state)

for (r1, r2) in test_rewards:
    total_r1 += r1
    total_r2 += r2
print("--------------------Total Test Reward------------------")
print("RL agent: {}, Other agent: {}".format(total_r1, total_r2))

################ Some Code Testing with RL Agent ####################
# sim1 = Simulation()
# agent1 = RL_agent(initial_coop=0.5)
# env = gym.make('prisoner_dilemma:prisoner-dilemma-v0', 
#                             enemy_agent=AutomatonAgent(strategy=sim1.random_agent),
#                             action_map=sim1.action_map, 
#                             payoff=sim1.payoff)
# action_a1 = agent1.get_action()
# state1, reward, done, info = env.step(action_a1)
# print("state: {}, reward: {}, done: {}, info: {}".format(state1, reward, done, info))
# action_a2 = info['p1_noisy_action_num_profile'][1]

# print("My action before the update: {} with prob {}".format(action_a1,
# agent1.get_action_prob(action_a1, action_a2)))
# agent1.update(action_a1, action_a2)
# print("My action after update: {} with prob {}".format(agent1.get_action(action_a1, action_a2),
# agent1.get_action_prob(action_a1, action_a2)))

################ Some Testing Code Below w Only Automata #################
# sim1 = Simulation()
# agent1 = AutomatonAgent(strategy=sim1.random_agent)
# env = gym.make('prisoner_dilemma:prisoner-dilemma-v0', 
#                             enemy_agent=AutomatonAgent(strategy=sim1.random_agent),
#                             action_map=sim1.action_map, 
#                             payoff=sim1.payoff)
# action_a1 = agent1.get_action()
# state1, reward, done, info = env.step(action_a1)
# print("state: {}, reward: {}, done: {}, info: {}".format(state1, reward, done, info))
# need to update the state of the second agent outside of the environment

# okay, yes, this for now will only assume that the RL agent learns against an automaton agent
# later version will allow RL agent to learn against a different RL agent, so I guess they will be 
# updated simultaneously


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