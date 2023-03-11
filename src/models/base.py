from abc import abstractmethod, ABCMeta

class ModelInterface(metaclass=ABCMeta): 
    def train(self, data: tuple):
        raise NotImplementedError('Training is not implemented yet')
    
    def test(self, data): 
        raise NotImplementedError('Testing is not implemented yet')
    