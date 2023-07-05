This code is associated with the following paper: 

```
@article{Kit_2023,
doi = {10.1088/1361-6587/acb3f7},
url = {https://dx.doi.org/10.1088/1361-6587/acb3f7},
year = {2023},
month = {feb},
publisher = {IOP Publishing},
volume = {65},
number = {4},
pages = {045003},
author = {A Kit and A E Järvinen and L Frassinetti and S Wiesen and JET Contributors},
title = {Supervised learning approaches to modeling pedestal density},
journal = {Plasma Physics and Controlled Fusion},
}
```


In order to reproduce the table from the paper, you will 
- Data 
	- If you have EUROfusion/JET credentials, please contact the authors for location of the jet pedestal database on Heimdall. 
	- Save this to `data/` which is one level above `src/`
- Instal dependencies from `requirements.txt`
- run searches for all models and all input spaces (**TBD**)
    - `chmod +x ./search_all_spaces.sh`
    - `./search_all_spaces.sh`
    - This produces a `.csv` for each model and input space combination in the `results` directory. 
- testing models 
	- Using the optimal parameters from the above search, set the base configuration for the model of choice in `src/configs/{model_name}_base_config.yaml`, and run `cv_train_test.py`. 
	- this will print to stdout the MSE, R2, MAP, etc., metrics for the given configuration
	- additionally it will plot and save a resulting figure to `/results/base_results/figures`

