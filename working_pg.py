import numpy as np
import random
import gym
import tensorflow as tf 
import matplotlib.pyplot as plt
from agents import AutomatonAgent, RL_agent, Simulation
#%matplotlib inline 

dims = 5
outs = 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim = dims, activation='relu'))
model.add(tf.keras.layers.Dense(outs, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
#compute_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make('CartPole-v0')
sim1 = Simulation()
max_num_rounds = 200
done = False
env = gym.make('prisoner_dilemma:prisoner-dilemma-v0', 
                            enemy_agent=AutomatonAgent(strategy=sim1.tft_agent),
                            action_map=sim1.action_map, 
                            payoff=sim1.payoff,
                            max_num_rounds=max_num_rounds,
                            noise=0)
episodes = 2000
scores = []
update_every = 5

gradBuffer = model.trainable_variables
for ix,grad in enumerate(gradBuffer):
  gradBuffer[ix] = grad * 0
  
for e in range(episodes):
  
  s = env.reset()
  
  ep_memory = []
  ep_score = 0
  done = False 
  while not done: 
    s = s.reshape([1,dims]).astype('float32')
    with tf.GradientTape() as tape:
      #forward pass
      logits = model(s)
    # our version
    #   a = 0 if random.random() <= logits else 1 # think more about these numbers
    #   logits = logits[0]
    # their version
      a_dist = logits.numpy()
    # Choose random action with p = action dist
      a = np.random.choice(a_dist[0],p=a_dist[0])
      a = np.argmax(a_dist == a)
      loss = compute_loss([a], logits)

    # make the choosen action 
    #s, r, done, info = env.step(a)
    #import pdb; pdb.set_trace();
    s, r, done, info = env.step(a, logits[0,a])
    print("RL action: {}, opponent action: {}".format(a, info["p1_noisy_action_profile"][1]))
    ep_score +=r
    if done: r-=10# MIGHT BE TOO EXTREME
    #import pdb; pdb.set_trace();
    #import pdb; pdb.set_trace();
    grads = tape.gradient(loss, model.trainable_variables)
    ep_memory.append([grads,r])
  scores.append(ep_score)
  # Discound the rewards 
  ep_memory = np.array(ep_memory)
  ep_memory[:,1] = discount_rewards(ep_memory[:,1])
  
  for grads, r in ep_memory:
    for ix,grad in enumerate(grads):
      gradBuffer[ix] += grad * r
  
  if e % update_every == 0:
    optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0
      
  if e % 100 == 0:
    print("--------------------------Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))