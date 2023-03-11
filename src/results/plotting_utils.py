import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import os 
if os.getenv('PLOTSTYLE') is not None: 
    plt.style.use(os.getenv('PLOTSTYLE'))
RED = "#dd3015"
GREEN = "#489A8C"
DARK = "#1C2C22"
GOLD = "#F87D16"
WHITE = "#FFFFFF"
BLUE = "#2E6C96"

def cv_train_test_result_plot(pred_nepeds: np.ndarray, test_neped: np.ndarray, model_name: str, input_space_idx: int, savefig=False): 
    mean_pred = np.mean(pred_nepeds, 0)
    std_pred = np.std(pred_nepeds, 0)

    fig = plt.figure(figsize=(10, 10))
    se = (test_neped - mean_pred)**2
    rmse = np.sqrt(sum(se) / len(mean_pred))
    plt.errorbar(test_neped, mean_pred, yerr=std_pred, color='grey', alpha=0.3, fmt='', linestyle='', zorder=20)
    plt.scatter(test_neped, mean_pred, edgecolors=(0, 0, 0), color=RED, s=100, label=f'RMSE: {rmse:.4}')
    lb, ub = 1, 12.5
    reg_x = np.linspace(lb, ub)
    plt.plot(reg_x, reg_x, lw=2, color='black')
    plt.plot(reg_x, reg_x*1.2, lw=2, color='black', ls='--')
    plt.plot(reg_x, reg_x*0.8, lw=2, color='black', ls='--')
    plt.xlabel('Experimental JET PDB $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.ylabel('Predicted $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)

    plt.legend(frameon=False)
    plt.title(f'{model_name} Input Space {input_space_idx}')
    if savefig: 
        if not os.path.exists('./results/base_results/figures/'): 
            os.mkdir('./results/base_results/figures/')
        plt.savefig(f"./results/base_results/figures/{model_name}_{input_space_idx}.svg", format='svg')
        plt.savefig(f"./results/base_results/figures/{model_name}_{input_space_idx}.png", dpi=300, transparent=True)
    plt.show()

def result_plot_with_outliers(pred_nepeds: np.ndarray, test_neped: np.ndarray, model_name: str, input_space_idx: int): 
    mean_pred = np.mean(pred_nepeds, 0)
    std_pred = np.std(pred_nepeds, 0)

    fig = plt.figure(figsize=(10, 10))
    se = (test_neped - mean_pred)**2
    high_se = np.argsort(se)
    rmse = np.sqrt(sum(se) / len(mean_pred))
    plt.errorbar(test_neped, mean_pred, yerr=std_pred, color='grey', alpha=0.3, fmt='', linestyle='', zorder=20)
    plt.scatter(test_neped, mean_pred, edgecolors=(0, 0, 0), color=RED, s=100, label=f'RMSE: {rmse:.4}')
    plt.scatter(test_neped[high_se[-10:]], mean_pred[high_se[-10:]], s=150, label='Top 10 Outliers', edgecolors=(0, 0, 0), color=WHITE)

    lb, ub = 1, 12.5
    reg_x = np.linspace(lb, ub)
    plt.plot(reg_x, reg_x, lw=2, color='black')
    plt.plot(reg_x, reg_x*1.2, lw=2, color='black', ls='--')
    plt.plot(reg_x, reg_x*0.8, lw=2, color='black', ls='--')
    plt.xlabel('Experimental JET PDB $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.ylabel('Predicted $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)

    plt.legend(frameon=False)
    plt.title(f'{model_name} Input Space {input_space_idx}\nOutliers')
    plt.savefig(f"./results/base_results/figures/{model_name}_{input_space_idx}_outliers.svg", format='svg')
    plt.savefig(f"./results/base_results/figures/{model_name}_{input_space_idx}_outliers.png", dpi=300, transparent=True)
    plt.show()

    pass 

def plot_neped_regression_quality(): 
    rel_dir = os.getenv('JETPDB')
    if rel_dir is None: 
        raise FileNotFoundError('Jet PDB not specified as an environment variable, please run the following command:\n  $export JETPDB=/path/to/jet-pedestal.csv')
    jet_pdb_all = pd.read_csv(rel_dir)
    jet_pdb = jet_pdb_all[(jet_pdb_all['elongation'] != -1) & (jet_pdb_all['Zeff'] != -1)]

    # Remove all all entries without an elongation, and Zeff
    jet_pdb = jet_pdb_all[(jet_pdb_all['elongation'] != -1) & (jet_pdb_all['Zeff'] != -1)]
    # If P_NBI or Gas puff is negative, set it to 0.0
    jet_pdb['P_NBI(MW)'][jet_pdb['P_NBI(MW)'] < 0] = 0.0 
    jet_pdb['gasflowrateofmainspecies10^22(e/s)'][jet_pdb['gasflowrateofmainspecies10^22(e/s)'] < 0] = 0.0 
    # Categorically format the divertor configuration, as it is of type string
    jet_pdb.loc[:, 'divertorconfiguration'] = jet_pdb.loc[:, 'divertorconfiguration'].astype('category')
    jet_pdb.loc[:, 'divertorconfiguration'] = jet_pdb.loc[:, 'divertorconfiguration'].cat.codes

    mp_array = jet_pdb[['Ip(MA)', 'averagetriangularity', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'gasflowrateofmainspecies10^22(e/s)', 'Meff']].to_numpy()
    def lorenzo(mps): 
        return 9.9*((mps[:, 0]**(1.24)) * (mps[:, 1]**(0.62)) * (mps[:, 2]**(-0.34))* (mps[:, 3]**(0.08))* (mps[:, 4]**(0.2)))
    real_neped = jet_pdb['nepedheight10^19(m^-3)'].to_numpy()
    pred_neped = lorenzo(mp_array)
    fig = plt.figure(figsize=(10, 10))
    lb, ub = 1, 12.5
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)
    plt.scatter(pred_neped, real_neped, edgecolors=(0, 0, 0), color=RED, s=100)
    reg_x = np.linspace(lb, ub)
    plt.plot(reg_x, reg_x, lw=2, color='black')
    plt.plot(reg_x, reg_x*1.2, lw=2, color=DARK, ls='--')
    plt.plot(reg_x, reg_x*0.8, lw=2, color=DARK, ls='--')
    plt.xlabel('Experimental JET PDB $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.ylabel('Predicted $n_{e, ped}$ ($10^{19}$ m$^{-3}$)')
    plt.title('Regression quality of log-linear model')
    plt.savefig(f"./results/LOR_FIT.png", transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    