import sys
sys.path.append('/home/gabe/.local/lib/python3.8/site-packages')

import numpy as np
import gym
import random
import time
import seaborn
import matplotlib.pyplot as plt
from random import randrange
import occupancy_env as env

def value_function(env, V, model, s, a, discount_rate):
    sum_vf = 0  # state value for state s
    for transition_probability, state, rewards, final_state in model[s][a]:     # see note #1 !
        # p  - transition probability from (s,a) to (s')
        # s_ - next state (s')
        # r  - reward on transition from (s,a) to (s')
        next_s, obs, rewards, done, info = env.step(s,a)
        sum_vf += transition_probability * (rewards + discount_rate * V[next_s])
    return sum_vf

def policy_iteration(env, action_space_size,state_space_size,model, gamma, theta):
    
    V = np.zeros(state_space_size)
    pi = np.zeros(state_space_size,dtype=int)

    count_evaluations = 0
    count_improvements = 0

    while True:
    
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(state_space_size):
                v = V[s]
                V[s] = value_function(env, V, model, s, pi[s], gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < theta: break
            count_evaluations +=1

        # 3. Policy Improvement
        policy_stable = True
        for s in range(state_space_size):
            old_action = pi[s]
            pi[s] = np.argmax([value_function(env, V, model, s, a, gamma)
                            for a in range(action_space_size)])
            if old_action != pi[s]: policy_stable = False

        if policy_stable: break
        count_improvements +=1

    print("Evaluations:" + str(count_evaluations))
    print("Improvements: " + str(count_improvements))    
    print(V)
    
    return V, pi