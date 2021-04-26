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


INITIAL_OCCUPANCY = 0
CURRENT_OCCUPANCY = 0
FINAL_OCCUPANCY = 0
MAX_PEOPLE_ALLOWED = 100


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.DEBUG):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, "w")        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

results_logger = setup_logger('results', 'results.log')

class OccupancyEnv(gym.Env):
    """An occupancy enhancer for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(OccupancyEnv, self).__init__()

        self.df = df
        self.reward_range = (0, 10)

        # Actions are either you count the detection or not.
        self.action_space = 2

        # Total detections throughout a day
        self.observation_space = len(df.index)

        self.P = {s: {a: [] for a in range(self.action_space)} for s in range(self.observation_space)}

        #if(self.total_occupancy == self.max_people_allowed):

        self.highest_occupancy = 0
        self.occupancy_trace = []

        for detection in range(self.observation_space):
            for actions in range(self.action_space):
                li = self.P[detection][actions]
                if detection == self.observation_space-1:
                    done = True
                    prob = 1.0
                else:
                    done = False
                    prob = 0.50
                li.append((prob, detection, 0, done))


    def _occupancy_factor(self, observations_so_far, occupancy, event):

        if occupancy > self.highest_occupancy:
            self.highest_occupancy = occupancy

        observation_rate =  ((self.observation_space) - observations_so_far) / (self.observation_space)

        midpoint = (self.observation_space-1) / 2
        sunset = midpoint + (midpoint / 2)

        if observations_so_far < midpoint:
            if event =='Enter': 
                bias = observation_rate
            if event =='Exit':
                bias = 1 - observation_rate

        if observations_so_far >= midpoint and observations_so_far < sunset:
            if event =='Enter': 
                bias = observation_rate
            if event =='Exit':
                bias = observation_rate

        if observations_so_far >= sunset:
            if event =='Enter': 
                bias = observation_rate * -1
            if event =='Exit':
                bias = observation_rate
        
        factor = bias 

        return factor


    def _next_observation(self):
        # Ideally, it should get the detection data for at least 5 frames before and after
        # For the purpose of this excersise, we will go with a single detection

        frame = np.array([self.df.loc[self.current_step: self.current_step, 'confidence'].values,
                          self.df.loc[self.current_step: self.current_step, 'event'].values,])

        # Ideally, we can append as much info as available
        # For now, let's just append the event type to the frame

        obs = np.array([[self.total_occupancy - INITIAL_OCCUPANCY], [MAX_PEOPLE_ALLOWED - self.total_occupancy]])

        obs = np.append(frame,obs)

        return obs

    def _take_action(self, action):

        event = self.df.loc[self.current_step, "event"]

        if action == 1:
            if event == 'Enter':
                self.total_occupancy += 1
            else:
                self.total_occupancy -= 1


    def step(self, s, action):
        # Execute one time step within the environment
        done = False
        reward = 0
        random.seed(19)
        confidence_score = self.df.loc[self.current_step, 'confidence']
        event = self.df.loc[self.current_step, "event"]

        if self.current_step < self.observation_space:
            self._take_action(action)
            self.current_step = s
        
        if (self.total_occupancy > self.max_people_allowed) or (self.total_occupancy < 0):
                done = True
                print("ouch, either you are counting negative or reaching max: " + str(self.current_step))
                reward = -1

                if self.total_occupancy < 0:
                    self.total_occupancy = 0
        
                if self.total_occupancy > self.max_people_allowed:
                    self.total_occupancy = MAX_PEOPLE_ALLOWED
        else:
            if action == 1:
                print("yay, moving on!")

                reward = self._occupancy_factor(self.current_step, self.total_occupancy, event)

                if self.current_step==(self.observation_space - 1):
                    done = True
                
            else:

                reward = 0

        obs = self._next_observation()


        return self.current_step, obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.initial_occupancy = INITIAL_OCCUPANCY
        self.final_occupancy = FINAL_OCCUPANCY
        self.max_people_allowed = MAX_PEOPLE_ALLOWED
        self.total_occupancy = INITIAL_OCCUPANCY

        # Set the current step to the beginning of the day
        self.current_step = 0

        return self._next_observation()

    def render(self, table, clemode='human', close=False):
        # Render the environment to the screen

        occupancy = self.initial_occupancy
        power_hours = []

        for i in range(len(table)):
                timestamp = self.df.loc[i, 'detection_timestamp']
                event = self.df.loc[i, "event"]
                row = np.array(table[i])
                biggest = np.argmax(row, axis = 0)
                if biggest == 1:
                    if event == 'Enter':
                            occupancy +=1
                    else:
                            occupancy -=1
                results_logger.debug('%s %s %s', str(timestamp)[0:19],",",occupancy)
                power_hours.append([timestamp, occupancy])
        
        POWER_HOURS = np.array(power_hours)
        np.set_printoptions(threshold=sys.maxsize)



