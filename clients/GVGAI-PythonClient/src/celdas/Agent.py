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
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Input, Reshape

tf.compat.v1.enable_v2_behavior()


MEMORY_CAPACITY = 10000
NUM_ACTIONS = 5
BATCH_SIZE = 30
GAMMA = 0.6
TAU = 0.08

elementToFloat = {
    'w': 1.0,
    'A': 2.0,
    'S': 3.0,
    'L': 4.0,
    'e': 5.0,
    'x': 6.0,
}


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
        # networkOptions = [
        #     # keras.layers.InputLayer(24, input_dim=117, activation='relu'),
        #     keras.layers.InputLayer(input_shape=(9,13), ),
        #     keras.layers.Dense(32, activation='softmax', kernel_initializer='random_uniform'),
        #     keras.layers.Dense(NUM_ACTIONS)
        # ]

        # self.policyNetwork = keras.Sequential(networkOptions)
        # self.targetNetwork = keras.Sequential(networkOptions)
        self.policyNetwork = self._build_compile_model()
        self.targetNetwork = self._build_compile_model()
        print(self.policyNetwork.summary())
        self.align_target_model()

    def _build_compile_model(self):
        # model = Sequential()
        # model.add(InputLayer(input_shape=(9,13)))
        # model.add(InputLayer(input_shape=(9,13), name='state'))
        # model.add(Input(shape=[9,13]),)
        # model.add(Dense(100, activation='relu', name='hiddenI'))
        # model.add(Dense(50, activation='relu', name='hiddenII'))
        # model.add(Dense(NUM_ACTIONS, activation='linear', name='actions'))
        
        inputs = Input(shape=(9,13), name='state')
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(10, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Zelda')

        model.compile(loss='mse') #, optimizer=self._optimizer)
        return model

    def align_target_model(self):
        self.targetNetwork.set_weights(self.policyNetwork.get_weights())
   

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

        currentPosition = self.getAvatarCoordinates(sso)

        if self.lastState is not None:
            reward = self.getReward(self.lastState, currentPosition)
            # print(reward)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
            self.train()
        # pprint(vars(sso))
        # print(self.get_perception(sso))

        self.lastState = sso

        # q_values = self.policyNetwork.predict(currentState)
        # return np.argmax(q_values[0])

        if sso.gameTick == 1000:
            return "ACTION_ESCAPE"
        else:
            index = random.randint(0, len(sso.availableActions) - 1)
            self.lastAction = index
            return sso.availableActions[index]

    def train(self):
        batch = self.replayMemory.sample(BATCH_SIZE)
        if len(batch) < BATCH_SIZE:
            return        

        for experience in batch:

            tensorState = tf.convert_to_tensor([self.get_perception(experience.state)])
            tensorNextState = tf.convert_to_tensor([self.get_perception(experience.nextState)])

            # Intentamos predecir la mejor accion
            target = self.policyNetwork.predict(tensorState)

            t = self.targetNetwork.predict(tensorNextState)
            # Para la accion que hicimos corregimos el Q-Value
            target[0][experience.action] = experience.reward + GAMMA * np.amax(t)

            # Entrenamos con la prediccion vs la correccion
            self.policyNetwork.fit(tensorState, target, epochs=1, verbose=0)

        print('done training')


        # flatStates = [self.get_perception(sample.state) for sample in batch]
        # # print (flatStates)
        # tensorStates = tf.convert_to_tensor(flatStates)
        # # predict Q(s,a) given the batch of states
        # predicted_q = self.policyNetwork(tensorStates)
        # loss = self.policyNetwork.train_on_batch(tensorStates)
        # target_q = predicted_q.numpy()
        # # print target_q
        # batch_idxs = np.arange(BATCH_SIZE) # [0, 1, ...BATCH_SIZE]


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
        print("GAME OVER")
        self.gameOver = True
        if self.lastAction is not None:
            reward = self.getReward(self.lastState, self.getAvatarCoordinates(sso))
            if not sso.isAvatarAlive:
                reward = -500.0
                print ('Murio')
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
        return random.randint(0, 2)


    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [int(position[1]/10), int(position[0]/10)]

    def getReward(self, lastState, currentPosition):
        level = self.get_perception(lastState)
        col = currentPosition[0] # col
        row = currentPosition[1] # row
        reward = 0
        if level[col][row] == 'A':
            # Did not move
            # TODO add killed enemy reward
            # print ('Agent')
            reward = -1.0
        elif level[col][row] == 'L':
            # Found key
            print ('Found key')
            self.foundKey = True
            reward = 100.0
        elif level[col][row] == 'S' and self.foundKey:
            # Won
            reward = 500.0
        # elif level[col][row] == 'e':
        #     # Died
        #     print ('Died')
        #     reward = -50.0
        else:
            print ('No entro a nignuno')
            print level[col][row]

        # print 'level: '
        # print level[col][row]
        # print reward
        return reward


    def get_perception(self, sso):
        sizeWorldWidthInPixels= sso.worldDimension[0]
        sizeWorldHeightInPixels= sso.worldDimension[1];
        levelWidth = len(sso.observationGrid);
        levelHeight = len(sso.observationGrid[0]);
        
        spriteSizeWidthInPixels =  sizeWorldWidthInPixels / levelWidth;
        spriteSizeHeightInPixels =  sizeWorldHeightInPixels/ levelHeight;
        level = np.ndarray((levelHeight, levelWidth))
        level[:] = 0.0
        # level[:] = '.'
        avatar_observation = Observation()
        for ii in range(levelWidth):                    
            for jj in range(levelHeight):
                listObservation = sso.observationGrid[ii][jj]
                if len(listObservation) != 0:
                    aux = listObservation[len(listObservation)-1]
                    if aux is None: continue
                    level[jj][ii] = elementToFloat[self.detectElement(aux)]
    

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
        