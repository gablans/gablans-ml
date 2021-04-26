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
import occupancy_env as env

def value_iter(env, V, P, nA, nS, gamma_main):
    gamma = gamma_main
    new_V = [0]*nS
    policy = [0]*nS
    delta = 0

    for s in range(nS):
        old_value = new_V[s]
        new_v = [0]*nA
        for a in range(nA):
            for prob, next_s, reward, terminal in P[s][a]:
                score, obs, rewards, done, info = env.step(s,a)
                new_v[a] += prob * (rewards + gamma * V[next_s])

        new_V[s] = max(new_v)
        policy[s] = np.argmax(new_v)
        delta = max(delta, abs(old_value - new_V[s]))

    return new_V, policy