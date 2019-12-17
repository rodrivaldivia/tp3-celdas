import random
import os
import datetime as dt
from time import sleep

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


MEMORY_CAPACITY = 30
# TIMESTEPS_PER_EPISODE = 150
STEPS_TO_UPDATE_NETWORK = 10
NUM_ACTIONS = 5
BATCH_SIZE = 5
GAMMA = 0.8

elementToFloat = {
    '.': 0.0,
    'w': 1.0,
    'A': 2.0,
    'S': 3.0,
    'L': 4.0,
    'e': 5.0,
    'x': 6.0,
}

directions = {
    'ACTION_DOWN':  (1,2),
    'ACTION_UP':    (1,0),
    'ACTION_RIGHT': (2,1),
    'ACTION_LEFT':  (0,1)
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
        self.exploreNext = False


    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    * @param sso Phase Observation of the current game.
    * @param elapsedTimer Timer (1s)
    """

    def init(self, sso, elapsedTimer):   
        self.lastState = None
        self.lastAction = None
        self.timestep = 0
        self.align_target_model()
        self.foundKey = False
        self.switchedDirection = False
        self.currentDirection = 'ACTION_DOWN'
        # Set KEY as goal
        # print(self.get_perception(sso))
        self.goalPosition = None
        for observations in sso.immovablePositions:
            if observations[0].itype == 4:
                self.goalPosition = observations[0].getPositionAsArray()
        # self.goalPosition = sso.immovablePositions[0][0].getPositionAsArray()
        print(self.goalPosition)


    def _build_compile_model(self):
        # inputs = Input(shape=(9,13), name='state')
        inputs = Input(shape=(3, 3, 3), name='state')
        x = Flatten()(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(NUM_ACTIONS, activation='relu')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Zelda')

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
        return model

    def align_target_model(self):
        self.targetNetwork.set_weights(self.policyNetwork.get_weights())
   

    def get_new_direction(self, action):
        # print('New action: ', action)
        # print('Old action: ',self.currentDirection)
        # print(action == 'ACTION_USE')
        self.switchedDirection = action != self.currentDirection and action != 'ACTION_USE'
        # print(self.switchedDirection)
        return self.currentDirection if action == 'ACTION_USE' else action 

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
        self.train()
        if sso.gameTick % STEPS_TO_UPDATE_NETWORK == 0:    
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

        if self.movementStrategy.shouldExploit() and not self.exploreNext:

            # print('Using strategy...')
            q_values = self.policyNetwork.predict(tensorState)
            index = np.argmax(q_values[0])
            # print('q_values: ', q_values)
            # print(sso.availableActions)
            # print('Predicted Best: ', sso.availableActions[index])
            self.currentDirection = self.get_new_direction(sso.availableActions[index])
            # print('Current direction: ', self.currentDirection)
            return sso.availableActions[np.argmax(q_values[0])]
        else:
            self.exploreNext = False
            # print('Exploring...')
            if sso.gameTick == 3000:
                return "ACTION_ESCAPE"
            else:
                index = random.randint(0, len(sso.availableActions) - 1)
                self.lastAction = index
                # print('Exploring: ', sso.availableActions[index])
                self.currentDirection = self.get_new_direction(sso.availableActions[index])
                # print('Current direction: ', self.currentDirection)
                return sso.availableActions[index]

    def train(self):
        # if self.starting:
        batch = self.replayMemory.sample(BATCH_SIZE)
        if len(batch) < BATCH_SIZE:
            return
        # else:
        # batch = self.replayMemory.popExperience()
        
        # print('start training')

        i = 0

        for experience in batch:

            tensorState = tf.convert_to_tensor([self.get_perception(experience.state)])
            tensorNextState = tf.convert_to_tensor([self.get_perception(experience.nextState)])

            # Intentamos predecir la mejor accion
            target = self.policyNetwork.predict(tensorState)

            t = self.targetNetwork.predict(tensorNextState)
            # Para la accion que hicimos corregimos el Q-Value
            # print('Policy prediction: ', target)
            # print('Target prediction: ', t)
            # print('Q value before: ', target[0][experience.action])
            # print('Experience reward: ', experience.reward)
            # print('Target max: ', np.amax(t))
            target[0][experience.action] = experience.reward + GAMMA * np.amax(t)
            # print('Q value after: ', target[0][experience.action])

            # Entrenamos con la prediccion vs la correccion
            self.policyNetwork.fit(tensorState, target, epochs=1, verbose=0)

        # print('done training')


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
        return [0]
        print('return to lvl random')


    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [int(position[1]/10), int(position[0]/10)]

    def getReward(self, lastState, currentState, currentPosition):
        level = self.get_level_perception(lastState)
        col = currentPosition[0] # col
        row = currentPosition[1] # row
        deltaDistance = self.getDistanceToGoal(self.getAvatarCoordinates(lastState)) - self.getDistanceToGoal(self.getAvatarCoordinates(currentState))
        reward = 10.0*(deltaDistance)

        moved = deltaDistance != 0

        if currentState.NPCPositionsNum < lastState.NPCPositionsNum:
            print('KILLED AN ENEMY')
            return 500.0

        if not moved:
            # Did not move
            print('DID NOT MOVE')
            print(currentState.availableActions[self.lastAction])
            if self.lastAction is not None and currentState.availableActions[self.lastAction] == 'ACTION_USE':
                print('BUT DID ATTACK')
                reward = -10.0
            else:
                if(self.switchedDirection):
                    print('SWITCHED DIRECTION')
                    reward = 0.0
                else:
                    print ('STEPPED INTO WALL')
                    reward = -50.0
        # elif level[col][row] == elementToFloat['.']:
            # print ('MOVED')
            # print (self.getDistanceToGoal(currentState))
        elif level[col][row] == elementToFloat['L']:
            print ('FOUND KEY')
            # Found key
            self.foundKey = True
            # Set GATE as new goal
            self.goalPosition = currentState.portalsPositions[0][0].getPositionAsArray()
            reward = 1000.0
        elif level[col][row] == elementToFloat['S'] and self.foundKey:
            # Won
            print('WON')
            reward = 5000.0
        # else:
        #     print ('No entro a nignuno')

        # print 'level: '
        # print level[col][row]
        print reward
        return reward

    def getDistanceToGoal(self, coordinates):
        # print('Getting distance to goal')
        # print(self.getAvatarCoordinates(state))
        # print(self.goalPosition)
        return distance.cityblock(coordinates, self.goalPosition)

    # def isCloserToKey(self, lastState, currentState):
    #     closer = self.getDistanceToKey(currentState) < self.getDistanceToKey(lastState)
    #     self.closerToKey = closer
    #     return closer

    # def isCloserToExit(self, lastState, currentState):
    #     closer = self.getDistanceToExit(currentState) < self.getDistanceToExit(lastState)
    #     self.closerToExit = closer
    #     return closer


    def get_level_perception(self, sso):
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

    def get_perception(self, sso):
        avatarPosition = self.getAvatarCoordinates(sso)
        level = np.ndarray((3,3))
        distances = np.ndarray((3,3))
        direction = np.ndarray((3,3))

        level[:] = 0.0
        distances[:] = 20
        direction[:] = 0.0
        direction[directions[self.currentDirection]] = 1.0

        # level[:] = '.'
        avatar_observation = Observation()
        for ii in range(3):                   
            for jj in range(3):
                iiPosition = ii + avatarPosition[1] - 1 
                jjPosition = jj + avatarPosition[0] - 1 
                listObservation = sso.observationGrid[iiPosition][jjPosition]
                if len(listObservation) != 0:
                    # print([jjPosition, iiPosition])
                    # print(self.getDistanceToGoal([int(jjPosition), int(iiPosition)]))
                    distances[jj][ii] = self.getDistanceToGoal([int(jjPosition), int(iiPosition)])
                    aux = listObservation[len(listObservation)-1]
                    if aux is None: continue
                    level[jj][ii] = elementToFloat[self.detectElement(aux)]
        # print(level)
        return [level, distances, direction]


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
        