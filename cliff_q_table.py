import numpy as np
import gym
import random
import time
env = gym.make('CliffWalking-v0').env
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# Hyperparameters
alpha = 0.1
gamma = 1.
epsilon = 0.0

N_episode = 1001

for i in range(1, N_episode):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    ret = 0
    while not done:
        if random.uniform(0, 1) < epsilon*(1-i/N_episode):
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        next_state, reward, done, info = env.step(action)    
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        state = next_state
        if ret<-100:
            break
