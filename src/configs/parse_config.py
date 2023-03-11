import yaml 
from pathlib import Path
import argparse 

from data import input_spaces
def parse_config(c_arg: argparse.Namespace) -> argparse.Namespace:
    config = yaml.safe_load(Path(f'./configs/base/{c_arg.config_file_name}.yaml').read_text())    
    args = argparse.Namespace(**config)
    
    if c_arg.input_space is not None: 
        args.trainer['input_space'] = c_arg.input_space
    model_name = args.model['object'].split('Interface')[0]
    input_space_idx = args.trainer['input_space']

    numerical_features = input_spaces[str(args.trainer['input_space'])]['numerical']
    categorical_features = input_spaces[str(args.trainer['input_space'])]['categorical']

    if (model_name in ['CatBoost', 'FFNN_CAT', 'TABNET', 'CATEMBED', 'NODE', 'AUTOINT']) and (input_space_idx in [1, 2, 3]):     
        args.model['cat_features'] = categorical_features
        args.model['num_features'] = numerical_features
    elif (model_name in ['CatBoost', 'FFNN_CAT', 'TABNET', 'CATEMBED', 'NODE', 'AUTOINT']) and (input_space_idx in [4,5,6]): 
        args.model['num_features'] = numerical_features
        if 'cat_features' in args.model.keys(): 
            args.model.pop('cat_features')

    mp_dim = len( numerical_features + categorical_features)
    args.model['mp_dim'] = mp_dim

    args.trainer['num_cv'] = c_arg.num_cv
    return args

def parse_search_config(c_arg: argparse.Namespace) -> argparse.Namespace: 
    config = yaml.safe_load(Path(f'./configs/search/{c_arg.config_file_name}.yaml').read_text())    
    args = argparse.Namespace(**config)
    return args

from typing import Union 
def update_local_search_config(config: dict, model_name: str, input_space: Union[str, int]):
    if model_name == 'CATEMBED': 
       config['layers'] = parse_layers_catembed(config) 

    numerical_features = input_spaces[str(input_space)]['numerical']
    categorical_features = input_spaces[str(input_space)]['categorical']

    if (model_name in ['CatBoost', 'FFNN_CAT', 'TABNET', 'CATEMBED', 'NODE', 'AUTOINT']) and (int(input_space) in [1, 2, 3]):     
        config['cat_features'] = categorical_features
        config['num_features'] = numerical_features
    elif (model_name in ['CatBoost', 'FFNN_CAT', 'TABNET', 'CATEMBED', 'NODE', 'AUTOINT']) and (int(input_space) in [4,5,6]): 
        config['num_features'] = numerical_features
        if 'cat_features' in config.keys(): 
            config.pop('cat_features')
    return config 

def parse_search_space_args(search_dict, model_name):
    from ray import tune 
    search_space = {}
    for param_name, param_dict in search_dict.items():
        if isinstance(param_dict, str) or isinstance(param_dict, int) or isinstance(param_dict, float): 
            search_space[param_name] = param_dict
            continue  
        elif isinstance(param_dict, list): 
            search_space[param_name] = [getattr(tune, item['dtype'])(item['l'], item['u']) for item in param_dict]    
            continue 
        param_dtype_ = param_dict['dtype']
        upper_bound, lower_bound = param_dict['u'], param_dict['l']
        search_space[param_name] = getattr(tune, param_dtype_)(lower_bound, upper_bound)
    
    if model_name == 'CATEMBED': 
        print(model_name)
        search_space['layers'] = tune.choice(
            [[tune.randint(8, 129) for _ in range(2)],
            [tune.randint(8, 259) for _ in range(3)],
            [tune.randint(8, 259) for _ in range(4)],
            [tune.randint(8, 513) for _ in range(5)],
            [tune.randint(8, 513) for _ in range(6)],
            [tune.randint(8, 513) for _ in range(7)],
            [tune.randint(8, 513) for _ in range(8)],
            ]
        )
    return search_space
def parse_layers_catembed(config):
    layers = []
    for i in range(len(config['layers'])): 
        layers.append(config['layers'][i])
    return '-'.join(str(x) for x in layers)