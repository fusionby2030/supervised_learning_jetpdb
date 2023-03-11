
input_spaces = {
    '1': {'numerical': ['Ip(MA)', 'B(T)', 'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity'], # 'uppertriangularity', 'lowertriangularity'],
                         'categorical': ['FLAG:Kicks', 'FLAG:RMP', 'FLAG:pellets', 'Atomicnumberofseededimpurity',
                                         'divertorconfiguration']
                         },
    '2': {'numerical': ['Ip(MA)', 'B(T)',   'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity', 'BetaN(MHD)'],
                                'categorical': ['FLAG:Kicks', 'FLAG:RMP', 'FLAG:pellets',
                                                'Atomicnumberofseededimpurity', 'divertorconfiguration']
                                },
    '3': {'numerical': ['Ip(MA)', 'B(T)',   'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity', 'BetaN(MHD)', 'Zeff'],
                                'categorical': ['FLAG:Kicks', 'FLAG:RMP', 'FLAG:pellets',
                                                'Atomicnumberofseededimpurity', 'divertorconfiguration']
                                },
    '4': {'numerical': ['Ip(MA)', 'B(T)',   'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity'],
                                  'categorical': []
                                  },
    '5': {'numerical': ['Ip(MA)', 'B(T)',   'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity', 'BetaN(MHD)'],
                                'categorical': []
                                },
    '6': {'numerical': ['Ip(MA)', 'B(T)',   'a(m)', 'P_NBI(MW)',
                                       'P_ICRH(MW)', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'plasmavolume(m^3)',
                                       'q95', 'gasflowrateofmainspecies10^22(e/s)', 'elongation', 'averagetriangularity', 'BetaN(MHD)', 'Zeff'],
                                'categorical': []
                                },
    '7': {'numerical': ['Ip(MA)', 'B(T)', 'averagetriangularity', 'P_NBI(MW)', 'gasflowrateofmainspecies10^22(e/s)'],
                         'categorical': []
                         },
    '8': {'numerical': ['Ip(MA)', 'averagetriangularity', 'P_TOT=PNBI+Pohm+PICRH-Pshi(MW)', 'gasflowrateofmainspecies10^22(e/s)', 'Meff'],
                         'categorical': []
                         },
    
}

import numpy as np 
import pandas as pd 
import os 
import warnings 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # IGNORE TENSORFLOW WARNINGS 
rel_dir = os.getenv('JETPDB')
if rel_dir is None: 
    raise FileNotFoundError('Jet PDB not specified as an environment variable, please run the following command:\n  $export JETPDB=/path/to/jet-pedestal.csv')
jet_pdb_all = pd.read_csv(rel_dir)
# Remove all all entries without an elongation, and Zeff
jet_pdb = jet_pdb_all[(jet_pdb_all['elongation'] != -1) & (jet_pdb_all['Zeff'] != -1)] # & (jet_pdb_all['P_NBI(MW)'] >= 0.0)& (jet_pdb_all['gasflowrateofmainspecies10^22(e/s)'] >= 0.0)
# If P_NBI or Gas puff is negative, set it to 0.0
jet_pdb['P_NBI(MW)'][jet_pdb['P_NBI(MW)'] < 0] = 0.0 
jet_pdb['gasflowrateofmainspecies10^22(e/s)'][jet_pdb['gasflowrateofmainspecies10^22(e/s)'] < 0] = 0.0 
# Categorically format the divertor configuration, as it is of type string
jet_pdb.loc[:, 'divertorconfiguration'] = jet_pdb.loc[:, 'divertorconfiguration'].astype('category')
jet_pdb.loc[:, 'divertorconfiguration'] = jet_pdb.loc[:, 'divertorconfiguration'].cat.codes

def norm_mps(mps, mp_means, mp_stds): 
    return (mps - mp_means) / mp_stds

from verstack.stratified_continuous_split import scsplit
def get_cv_iterator(num_cv: int = 15, input_space_idx: int = 1):
    input_str = str(input_space_idx)
    rel_cols =  input_spaces[input_str]['numerical'] + input_spaces[input_str]['categorical']
    local_copy = jet_pdb.reset_index()
    X_all = local_copy[rel_cols]
    y_all = local_copy['nepedheight10^19(m^-3)']
    
    train_size, test_size = 0.8, 0.2
    X_trainval, X_test, y_trainval, y_test = scsplit(X_all, y_all, stratify=y_all, train_size=train_size, random_state=18)
    val_size = 0.25
    X_test_np, y_test_np =  X_test.to_numpy(), y_test.to_numpy()
    X_trainval, y_trainval = X_trainval.reset_index(drop=True), y_trainval.reset_index(drop=True)
    
    for rng in list(range(num_cv)):
        X_train, X_val, y_train, y_val = scsplit(X_trainval, y_trainval, stratify=y_trainval, test_size=val_size, random_state=rng)
        train_mps_np, val_mps_np = X_train.to_numpy(), X_val.to_numpy(),
        train_nepeds_np, val_nepeds_np = y_train.to_numpy(), y_val.to_numpy()
        test_mps_np = X_test_np.copy()
        if input_space_idx in [1,2,3]: 
            # Can only normalize the numerical features in this case
            num_numerical = len(input_spaces[input_str]['numerical']) 
            mp_means, mp_stds = np.mean(train_mps_np, axis=0), np.std(train_mps_np, axis=0)            
            train_mps_np[:, :num_numerical], val_mps_np[:, :num_numerical], test_mps_np[:, :num_numerical] = norm_mps(train_mps_np[:, :num_numerical], mp_means[:num_numerical], mp_stds[:num_numerical]), norm_mps(val_mps_np[:, :num_numerical], mp_means[:num_numerical], mp_stds[:num_numerical]), norm_mps(test_mps_np[:, :num_numerical], mp_means[:num_numerical], mp_stds[:num_numerical])
        else: 
            mp_means, mp_stds = np.mean(train_mps_np, axis=0), np.std(train_mps_np, axis=0)
            train_mps_np, val_mps_np, test_mps_np = norm_mps(train_mps_np, mp_means, mp_stds), norm_mps(val_mps_np, mp_means, mp_stds), norm_mps(test_mps_np, mp_means, mp_stds)
        yield (train_mps_np, train_nepeds_np), (val_mps_np, val_nepeds_np), (test_mps_np, y_test_np)

from torch.utils.data import Dataset

class TorchDataset(Dataset): 
    def __init__(self, X, y): 
        self.X, self.y = X, y 
    
    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]
