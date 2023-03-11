from .base import ModelInterface

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

class RandomForestInterface(ModelInterface): 
    def __init__(self, params): 
        if 'mp_dim' in params.keys(): 
            params.pop('mp_dim')
        self.params = params 

    def train(self, data):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

        params = self.params 
        
        self.rf = RandomForestRegressor(**params)
        self.rf.fit(X_train, y_train)

    def test(self, data): 
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

        test_pred = self.rf.predict(X_test)
        return y_test, test_pred
    

class ExtraTreesInterface(ModelInterface): 
    def __init__(self, params): 
        self.params = params 

    def train(self, data):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

        params = self.params 
        if 'mp_dim' in params.keys(): 
            params.pop('mp_dim')
        self.rf = ExtraTreesRegressor(**params)
        self.rf.fit(X_train, y_train)

    def test(self, data): 
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

        test_pred = self.rf.predict(X_test)
        return y_test, test_pred