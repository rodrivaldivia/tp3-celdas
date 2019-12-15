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
from scipy.spatial import distance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten

tf.compat.v1.enable_v2_behavior()

# tf.reset_default_graph()


MEMORY_CAPACITY = 100
TIMESTEPS_PER_EPISODE = 50
NUM_ACTIONS = 5
BATCH_SIZE = 30
GAMMA = 0.6
TAU = 0.08

elementToFloat = {
    '.': 0.0,
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
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0
        self.policyNetwork = self._build_compile_model()
        self.targetNetwork = self._build_compile_model()
        if self.episode == 0 and os.path.exists("./celdas/network/zelda.index"):
            self.policyNetwork.load_weights("./celdas/network/zelda")
        print(self.policyNetwork.summary())


    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    * @param sso Phase Observation of the current game.
    * @param elapsedTimer Timer (1s)
    """

    def init(self, sso, elapsedTimer):    
        print(sso.availableActions)
        self.lastState = None
        self.lastAction = None
        self.timestep = 0
        self.align_target_model()
        self.foundKey = False
        # Set KEY as goal
        print(self.get_perception(sso))
        self.goalPosition = None
        for observations in sso.immovablePositions:
            if observations[0].itype == 4:
                self.goalPosition = observations[0].getPositionAsArray()
        # self.goalPosition = sso.immovablePositions[0][0].getPositionAsArray()
        print(self.goalPosition)


    def _build_compile_model(self):
        inputs = Input(shape=(9,13), name='state')
        x = Flatten()(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='softmax')(x)
        outputs = Dense(NUM_ACTIONS, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Zelda')

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
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

        self.timestep += 1

        if sso.gameTick % TIMESTEPS_PER_EPISODE == 0:
            self.train()
            self.align_target_model()


        currentPosition = self.getAvatarCoordinates(sso)

        if self.lastState is not None:
            reward = self.getReward(self.lastState, sso, currentPosition)
            # print(reward)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
        
        # pprint(vars(sso))
        # print(self.get_perception(sso))

        self.lastState = sso


        tensorState = tf.convert_to_tensor([self.get_perception(sso)])

        if self.movementStrategy.shouldExploit():
            # print('Using strategy...')
            q_values = self.policyNetwork.predict(tensorState)
            print('q_values: ', q_values)
            print('Predicted Best: ', sso.availableActions[np.argmax(q_values[0])])
            return sso.availableActions[np.argmax(q_values[0])]
        else:
            # print('Exploring...')
            if sso.gameTick == 3000:
                return "ACTION_ESCAPE"
            else:
                index = random.randint(0, len(sso.availableActions) - 1)
                self.lastAction = index
                print('Exploring: ', sso.availableActions[index])
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
        self.episode += 1
        self.policyNetwork.save_weights("./celdas/network/zelda")
        print('Model saved!')
        if self.lastAction is not None:
            reward = self.getReward(self.lastState, sso, self.getAvatarCoordinates(sso))
            if not sso.isAvatarAlive:
                reward = -5000.0
                print ('AGENT KIA')
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
            self.train()
        # Two different levels
        # return random.randint(0, 2)
        print('return to lvl 0')
        return int(0)
        print('return to lvl random')


    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [int(position[1]/10), int(position[0]/10)]

    def getReward(self, lastState, currentState, currentPosition):
        level = self.get_perception(lastState)
        col = currentPosition[0] # col
        row = currentPosition[1] # row
        reward = 2.0*(self.getDistanceToGoal(lastState) - self.getDistanceToGoal(currentState))

        # VALIDAR QUE NO AtaQUE Y SI NO SE MUEVE TABLA

        if currentState.NPCPositionsNum < lastState.NPCPositionsNum:
            print('KILLED AN ENEMY')
            return 5000.0

        if level[col][row] == elementToFloat['A']:
            # Did not move
            # TODO add killed enemy reward
            if self.lastAction is not None and currentState.availableActions[self.lastAction] == 'ACTION_USE':
                # print('BUT DID ATTACK')
                reward = -6.0
            else:
                print ('AGENT DID NOT MOVE')
                reward -= 5.0
        elif level[col][row] == elementToFloat['.']:
            # Found key
            print ('MOVED')
            print (self.getDistanceToGoal(currentState))
        elif level[col][row] == elementToFloat['L']:
            # Found key
            print ('FOUND KEY')
            self.foundKey = True
            # Set GATE as new goal
            self.goalPosition = sso.portalsPositions[0][0].getPositionAsArray()
            reward = 10000.0
        elif level[col][row] == elementToFloat['S'] and self.foundKey:
            # Won
            print('WON')
            reward = 50000.0
        else:
            print ('No entro a nignuno')

        # print 'level: '
        # print level[col][row]
        print reward
        return reward

    def getDistanceToGoal(self, state):
        # print('Getting distance to goal')
        # print(self.getAvatarCoordinates(state))
        # print(self.goalPosition)
        return distance.cityblock(self.getAvatarCoordinates(state), self.goalPosition)

    # def isCloserToKey(self, lastState, currentState):
    #     closer = self.getDistanceToKey(currentState) < self.getDistanceToKey(lastState)
    #     self.closerToKey = closer
    #     return closer

    # def isCloserToExit(self, lastState, currentState):
    #     closer = self.getDistanceToExit(currentState) < self.getDistanceToExit(lastState)
    #     self.closerToExit = closer
    #     return closer


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
        