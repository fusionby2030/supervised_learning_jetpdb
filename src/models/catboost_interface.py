from .base import ModelInterface
import catboost as cb
import pandas as pd 

class CatBoostInterface(ModelInterface): 
    def __init__(self, params): 
        if 'mp_dim' in params.keys(): 
            params.pop('mp_dim')
        self.params: dict = params 

    def train(self, data):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        # eval_set = (X_val.tolist(), y_val)
        params = self.params.copy() 
        if 'cat_features' in params.keys(): 
            # create pandas dataframe  
            categorical_features = params['cat_features']
            numerical_features = params.pop('num_features')
            cols = numerical_features + categorical_features
            train_df = pd.DataFrame(X_train, columns=cols)
            val_df = pd.DataFrame(X_val, columns=cols)
            for col in categorical_features: 
                train_df.loc[:, col] = train_df.loc[:, col].astype('category')
                train_df.loc[:, col] = train_df.loc[:, col].cat.codes
                val_df.loc[:, col] = val_df.loc[:, col].astype('category')
                val_df.loc[:, col] = val_df.loc[:, col].cat.codes

            train_dataset = cb.Pool(train_df, y_train, cat_features=categorical_features)
            eval_dataset = cb.Pool(val_df, y_val, cat_features=categorical_features)
        else: 
            numerical_features = params.pop('num_features')
            cols = numerical_features
            train_df = pd.DataFrame(X_train, columns=cols)
            val_df = pd.DataFrame(X_val, columns=cols)
            train_dataset = cb.Pool(train_df, y_train)
            eval_dataset = cb.Pool(val_df, y_val)

        self.model = cb.CatBoostRegressor(**params, logging_level='Silent')
        self.model.fit(train_dataset, eval_set=eval_dataset)

    def test(self, data): 
        _, _, (X_test, y_test) = data
        params = self.params.copy()
        if 'cat_features' in params.keys(): 
            categorical_features = params['cat_features']
            numerical_features = params.pop('num_features')
            cols = numerical_features + categorical_features
            test_df = pd.DataFrame(X_test, columns=cols)
            for col in categorical_features: 
                test_df.loc[:, col] = test_df.loc[:, col].astype('category')
                test_df.loc[:, col] = test_df.loc[:, col].cat.codes
        else: 
            numerical_features = params.pop('num_features')
            cols = numerical_features
            test_df = pd.DataFrame(X_test, columns=cols)
        
        test_pred = self.model.predict(test_df)
        return y_test, test_pred