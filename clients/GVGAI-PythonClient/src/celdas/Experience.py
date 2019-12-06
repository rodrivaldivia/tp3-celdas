class Experience():
    def __init__(self, state, action, reward, nextState):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)