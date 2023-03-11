import argparse 
import numpy as np 
from configs import parse_config
from data import get_cv_iterator, calculate_metrics
import models 
import results 

parser = argparse.ArgumentParser('Config loc')
parser.add_argument('-c', '--config_file_name', type=str, help='Name of the config file to load')
parser.add_argument('-is', '--input_space', type=int, default=None, help='Input space')
parser.add_argument('-ncv', '--num_cv', type=int, default=15, help='Number of CV iterations to do')
parser.add_argument('-plot', '--plot_final_results', type=bool, default=False)
c_arg = parser.parse_args()
args = parse_config(c_arg)
model_name = args.model['object'].split('Interface')[0]
input_space_idx = args.trainer['input_space']
num_cv = args.trainer['num_cv']

model = getattr(models, args.model.pop('object'))(args.model)

print(f"\nTraining and evaluating {model_name} on Input Space {input_space_idx} for {num_cv} cross-validation folds")

cv_batches = get_cv_iterator(num_cv, input_space_idx)
_, _, (_, y) = next(cv_batches)
pred_nepeds = np.empty(shape=(num_cv, y.shape[0]))
cv_batches = get_cv_iterator(num_cv, input_space_idx)
cv_test_r2, cv_test_mse, cv_test_mape, cv_test_rmse = [0]*num_cv, [0]*num_cv, [0]*num_cv, [0]*num_cv

for k, batch in enumerate(cv_batches):  # (X_train, y_train), (X_val, ..) (X_test, )... 
    model.train(batch)
    test_neped, test_pred = model.test(batch)
    pred_nepeds[k] = test_pred 
    cv_test_results = calculate_metrics(test_neped, test_pred)
    cv_test_r2[k], cv_test_mse[k], cv_test_mape[k], cv_test_rmse[k] = cv_test_results[0], cv_test_results[1], cv_test_results[2], cv_test_results[3]
print('#############################')
print('###      Final Results    ###')
print(f'### MSE  {np.mean(cv_test_mse):.4f} +- {np.std(cv_test_mse):.4f} ###')
print(f'### RMSE {np.mean(cv_test_rmse):.4f} +- {np.std(cv_test_rmse):.4f} ###')
print(f'### R2   {np.mean(cv_test_r2):.4f} +- {np.std(cv_test_r2):.4f} ###')
print(f'### MAPE {np.mean(cv_test_mape):.4f} +- {np.std(cv_test_mape):.4f} ###')
print('#############################')

if c_arg.plot_final_results: 
    results.cv_train_test_result_plot(pred_nepeds, test_neped, model_name, input_space_idx)
