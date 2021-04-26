import sys
sys.path.append('/home/gabe/.local/lib/python3.8/site-packages')

import numpy as np
import gym
import random
import time
import seaborn
import matplotlib.pyplot as plt
from random import randrange

def learning_chart(rewards_from_all_episodes,num_episodes,rate):
    
    rewards_register = []
    rewards_per_thousand_episodes = np.split(np.array(rewards_from_all_episodes),num_episodes/rate)
    count = rate
    print("**Average reward per thousand episodes**\n")
    for r in rewards_per_thousand_episodes:
        rewards_register.append(sum(r/rate))
        print(count,":",str(sum(r/rate)))
        count += rate
        
    plt.plot(rewards_register)
    plt.ylabel('Rewards per 1000 episodes')
    plt.xlabel('Thousand episodes')
    plt.show(block=False)

def print_table(table_name, table):
    print("\n\n***** " + table_name +" *****\n")
    print(table)

def q_learning_algo(env, action_space_size,state_space_size,num_episodes, max_steps_per_episode, \
                    learning_rate, discount_rate, exploration_rate, max_exploration_rate, \
                    min_exploration_rate, exploration_decay_rate):
    
    rewards_all_episodes = []
    q_table = np.zeros((state_space_size, action_space_size))
    
    
    for episode in range(num_episodes):
        
        env.reset() 
        state = env.current_step
        done = False
        rewards_current_episode = 0
    
        # a Step is a single timestep within an episode
        for step in range(max_steps_per_episode):
            random.seed((step+1) * episode)
        
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                action = random.randint(0,1)
            
            new_state, obs, reward, done, info = env.step(step,action)
        
            #Update Q-Table for Q(s,a)
            q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
            state = new_state
            rewards_current_episode += reward
        
            if done == True:
                break
            
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

        rewards_all_episodes.append(rewards_current_episode)
    
    return(rewards_all_episodes, q_table)
    
    
