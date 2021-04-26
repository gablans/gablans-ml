
import sys
sys.path.append('/home/gabe/.local/lib/python3.8/site-packages')

import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
import seaborn
import matplotlib.pyplot as plt
import math

import pandas as pd
import datetime as dt
import occupancy_ql as ql
import occupancy_vi as vi
import occupancy_pi as pi

import occupancy_env as env
from occupancy_env import setup_logger
from datetime import datetime


def l2_norm(value_function, new_value_function):
    s = 0
    for i in range(len(value_function)):
        s += (value_function[i]-new_value_function[i])**2
    return math.sqrt(s)

def build_qtable(table):
    qt = []

    for i in range(len(table)-1):
        if table[i]==0:
            #qt[i]= [1,0]
            qt.append([1,0])
        else:
            qt.append([0,1])
            #qt[i]= [0,1]
    return qt

df = pd.read_csv('sampleday.csv')
df = df.sort_values('detection_timestamp')

#logging.basicConfig(filename='execution.log', level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger = setup_logger('main_run', 'logfile.log')

env = env.OccupancyEnv(df)
env.reset()

theta = 0.0001
gamma = 0.9
V = [0]*env.observation_space
policy = None

###################################################
# Choose 'option' between:                        #
#                                                 #
# 1. Value Iteration                              #
# 2. Policy Iteration                             #
# 3. Q Learning                                   #
###################################################

option = 1

###################################################

logger.debug("Starting: " + str(datetime.now()))
starting = datetime.now()

if option == 1:  # Value Iteration
    n_iter = 1000
    diff = []

    for i in range(n_iter):
        env.reset()
        new_V, policy = vi.value_iter(env, V, env.P, env.action_space, env.observation_space, gamma)
        delta = l2_norm(V, new_V)
        if delta < theta:
            break
        diff.append(delta)
        V = new_V
        logger.debug("Iteration: " + str(i) + "=======================")
        logger.debug("Delta: " + str(delta) )
        logger.debug("Value Function: " + str(V) )
        logger.debug("Policy: " + str(policy) )

        plt.plot(diff)
        plt.ylabel('V[k] - V[k-1]')
        plt.xlabel('Iterations until convergence')
        plt.savefig('occupancy_vi.png')

    qtable = build_qtable(policy)
    env.render(qtable)
    

if option == 2: # Policy Iteration

    env.reset()
    value, policy = pi.policy_iteration(env, env.action_space, env.observation_space, env.P, gamma, theta)
    qtable = build_qtable(policy)
    env.render(qtable)

if option == 3: # Q Learning

    env.reset()
    num_episodes = 20000
    max_steps_per_episode = env.observation_space
    learning_rate = 0.1
    gamma = 0.99
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    full_rewards, q = ql.q_learning_algo(env, env.action_space, env.observation_space, num_episodes, max_steps_per_episode, learning_rate, gamma, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate)
    ql.print_table("Q", q)
    env.render(q)

it_took = datetime.now() - starting
logger.debug("Ending: " + str(datetime.now()))
logger.debug("It took: " + str(it_took))
