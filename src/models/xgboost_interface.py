from .base import ModelInterface
import xgboost as xgb 

class XGBoostInterface(ModelInterface): 
    def __init__(self, params): 
        
        if 'mp_dim' in params.keys(): 
            params.pop('mp_dim')
        self.params = params 

    def train(self, data):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dval = xgb.DMatrix(X_val, label=y_val)
        self.dtest = xgb.DMatrix(X_test)
        self.y_test = y_test
        params = self.params 
        self.bst = xgb.train(params, self.dtrain, 50, [(self.dtrain, 'train'), (self.dval, 'eval')], verbose_eval=False)

    def test(self, batch=None): 
        test_pred = self.bst.predict(self.dtest)
        y_test = self.y_test
        return y_test, test_pred