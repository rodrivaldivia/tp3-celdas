import random
import os
import datetime as dt

from AbstractPlayer import AbstractPlayer
from Types import *

from EpsilonStrategy import EpsilonStrategy
from ReplayMemory import ReplayMemory
from Experience import Experience

from utils.Types import LEARNING_SSO_TYPE
from utils.SerializableStateObservation import Observation
import math
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.enable_v2_behavior()


MEMORY_CAPACITY = 10000
NUM_ACTIONS = 5
BATCH_SIZE = 6
GAMMA = 0.95
TAU = 0.08


class Agent(AbstractPlayer):
    def __init__(self):
        AbstractPlayer.__init__(self)
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
       

    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    * @param sso Phase Observation of the current game.
    * @param elapsedTimer Timer (1s)
    """

    def init(self, sso, elapsedTimer):    
        self.lastState = None
        self.lastAction = None
        networkOptions = [
            keras.layers.Dense(24, input_dim=117, activation='relu'),
            keras.layers.Dense(32, activation='softmax'),
            keras.layers.Dense(NUM_ACTIONS)
        ]

        # self.policyNetwork = keras.Sequential(networkOptions)
        # self.targetNetwork = keras.Sequential(networkOptions)

    """
     * Method used to determine the next move to be performed by the agent.
     * This method can be used to identify the current state of the game and all
     * relevant details, then to choose the desired course of action.
     *
     * @param sso Observation of the current state of the game to be used in deciding
     *            the next action to be taken by the agent.
     * @param elapsedTimer Timer (40ms)
     * @return The action to be performed by the agent.
     """

    def act(self, sso, elapsedTimer):

        # Hardcodded for now

        if self.lastState is not None:
            reward = 0 # get reward
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
        
        # pprint(vars(sso))
        print(self.get_perception(sso))



        self.lastState = sso

        if sso.gameTick == 1000:
            return "ACTION_ESCAPE"
        else:
            index = random.randint(0, len(sso.availableActions) - 1)
            self.lastAction = index
            return sso.availableActions[index]

    """
    * Method used to perform actions in case of a game end.
    * This is the last thing called when a level is played (the game is already in a terminal state).
    * Use this for actions such as teardown or process data.
    *
    * @param sso The current state observation of the game.
    * @param elapsedTimer Timer (up to CompetitionParameters.TOTAL_LEARNING_TIME
    * or CompetitionParameters.EXTRA_LEARNING_TIME if current global time is beyond TOTAL_LEARNING_TIME)
    * @return The next level of the current game to be played.
    * The level is bound in the range of [0,2]. If the input is any different, then the level
    * chosen will be ignored, and the game will play a random one instead.
    """

    def result(self, sso, elapsedTimer):
        return random.randint(0, 2)

    def get_perception(self, sso):
        sizeWorldWidthInPixels= sso.worldDimension[0]
        sizeWorldHeightInPixels= sso.worldDimension[1];
        levelWidth = len(sso.observationGrid);
        levelHeight = len(sso.observationGrid[0]);
        
        spriteSizeWidthInPixels =  sizeWorldWidthInPixels / levelWidth;
        spriteSizeHeightInPixels =  sizeWorldHeightInPixels/ levelHeight;
        level = np.chararray((levelHeight, levelWidth))
        level[:] = '.'
        avatar_observation = Observation()
        for ii in range(levelWidth):                    
            for jj in range(levelHeight):
                listObservation = sso.observationGrid[ii][jj]
                if len(listObservation) != 0:
                    aux = listObservation[len(listObservation)-1]
                    if aux is None: continue
                    level[jj][ii] = self.detectElement(aux)
    

        return level

    def detectElement(self, o):
        if o.category == 4:
            if o.itype == 3:
                return '0'
            elif o.itype == 0:
                return 'w'
            elif o.itype == 4:
                return 'L'
            else:
                return 'A'
            
             
        elif o.category == 0:
            if o.itype == 5:
                return 'A'
            elif o.itype == 6:
                return 'B'
            elif o.itype == 1:
                return 'A'
            else:
                return 'A'
             
        elif o.category == 6:
            return 'e'
        elif o.category == 2:
            return 'S'
        elif o.category == 3:
            if o.itype == 1:
                return 'e'
            else:         
                return 'e'         
        elif o.category == 5:
            if o.itype == 5:
                return 'x'
            else:         
                return 'e'
        else:                          
            return '?'
        