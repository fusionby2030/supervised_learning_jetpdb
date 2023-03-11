from .base import ModelInterface
from pytorch_tabular import TabularModel
from pytorch_tabular.models import AutoIntConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig 
import pandas as pd 
import numpy as np 

class AUTOINTInterface(ModelInterface): 
    def __init__(self, params): 
        self.params: dict = params 
        self.numerical_features = params['num_features']
        if 'cat_features' in self.params.keys():
            self.categorical_features = params['cat_features'] 
            self.pandas_cols = self.numerical_features + self.categorical_features
            self.data_confg = DataConfig(target=['target'], continuous_cols=self.params['num_features'], categorical_cols=self.categorical_features)
        else:
            self.categorical_features = None
            self.pandas_cols = self.numerical_features
            self.data_confg = DataConfig(target=['target'], continuous_cols=self.params['num_features'])
        
        self.trainer_config = TrainerConfig(auto_lr_find=False, batch_size=self.params['batch_size'], max_epochs=self.params['epochs'])
        self.optim_config = OptimizerConfig(optimizer=self.params['optimizer'])
        self.model_config =  AutoIntConfig(task='regression', target_range=[(0.1, 15)], learning_rate=self.params['learning_rate'])
    def train(self, data):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        
        self.tab_model = TabularModel(data_config=self.data_confg, 
                                 model_config=self.model_config, 
                                 optimizer_config=self.optim_config, 
                                 trainer_config=self.trainer_config)
                                 
        train_df = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], -1), columns=self.pandas_cols + ['target'])
        val_df = pd.DataFrame(np.concatenate([X_val, y_val.reshape(-1, 1)], -1), columns=self.pandas_cols + ['target'])

        self.tab_model.fit(train=train_df, validation=val_df)

    def test(self, data): 
        _, _, (X_test, y_test) = data
        test_df = pd.DataFrame(np.concatenate([X_test, y_test.reshape(-1, 1)], -1), columns=self.pandas_cols + ['target'])
        pred_df = self.tab_model.predict(test_df)
        return y_test, pred_df['target_prediction'].to_numpy() 