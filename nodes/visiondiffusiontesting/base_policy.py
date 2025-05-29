from abc import ABC, abstractmethod

# abstract base class
class Policy(ABC):
    @abstractmethod
    def train(self, states, actions):
        # states are list of nxT, actions are list of mxT where each list item is an episode
        pass

    @abstractmethod
    def save(self, location):
        # location is a fully qualified path for the learned model
        pass

    @abstractmethod
    def load(self, location):
        # location is a fully qualified path for the learned model
        pass

    @abstractmethod
    def getAction(self, state):
        # state is an array of length n, returns an action of length m
        pass

    @abstractmethod
    def forecastAction(self,state,num_seeds,length):
        # forecast is list (num seeds) of m x length
        pass

    @abstractmethod
    def reset(self):
        # for a new instance of the task in case any internals need to be reset
        pass