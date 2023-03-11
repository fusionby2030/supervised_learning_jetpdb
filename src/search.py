import argparse
from configs import parse_search_config, parse_search_space_args, update_local_search_config
from data import get_cv_iterator, calculate_metrics
import ray 
from ray import tune 
from ray.air import session 
from ray.tune.search import ConcurrencyLimiter 
from ray.tune.search.hyperopt import HyperOptSearch 
from hyperopt import hp 
from typing import List

import models 
def evaluate(model, cv_batch): 
    # Model training 
    model.train(cv_batch)
    test_neped, test_pred = model.test(cv_batch)
    cv_test_results = calculate_metrics(test_neped, test_pred)
    rmse_score = cv_test_results[3]
    return rmse_score 
    
def objective(config): 
    NUM_CV = 5
    config = update_local_search_config(config, model_name, input_space)
    model = getattr(models, interface_name)(config)
    cv_batches = get_cv_iterator(NUM_CV, input_space)

    mean_score = 0.0
    for k, batch in enumerate(cv_batches): 
        batch_score = evaluate(model, batch)
        mean_score += batch_score
    session.report({'mean_loss': mean_score / (k + 1)})


parser = argparse.ArgumentParser('Config loc')
parser.add_argument('-c', '--config_file_name', type=str, help='Name of the search config file to load')
parser.add_argument('-is', '--input_space', type=int, help='The search space to use, as defined in paper, can take on values 1-8', default=1)
parser.add_argument('-nit', '--num_iterations', type=int, help='Number of search trials to run', default=10)
file_arg = parser.parse_args()
args = parse_search_config(file_arg)

ray.init(configure_logging=False)
interface_name = args.model['object']
model_name = args.model['object'].split('Interface')[0]
input_space = file_arg.input_space 
num_samples = int(file_arg.num_iterations)


initial_params: List[dict] = [args.initial_params] if model_name != 'CATEMBED' else None
algo = HyperOptSearch(points_to_evaluate=initial_params)
algo = ConcurrencyLimiter(algo, max_concurrent=4)
search_config = parse_search_space_args(args.search_space, model_name)

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_config,
)
results = tuner.fit()
best_hparam_dict = results.get_best_result().config
result_df = results.get_dataframe()
best_result = min(result_df['mean_loss'])
print("Best hyperparameters found were: ", best_hparam_dict)
print('with result', best_result)


result_df.to_csv(f'./results/search_results/{model_name}_{file_arg.input_space}_results_df.csv')
ray.shutdown()